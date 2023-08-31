public class Main {
public static void count_setbit ( int N ) {
int result = 0 ;
for ( int i = 0 ; i < 32 ; i ++ ) {
if ( ( ( 1 << i ) & N ) > 0 ) {
result ++ ; } } System . out . println ( result ) ; }
public static void main ( String [ ] args ) { int N = 43 ; count_setbit ( N ) ; } }
import java . lang . Math ; class GFG {
public static boolean isPowerOfTwo ( int n ) { return ( Math . ceil ( Math . log ( n ) / Math . log ( 2 ) ) == Math . floor ( Math . log ( n ) / Math . log ( 2 ) ) ) ; }
public static void main ( String [ ] args ) { int N = 8 ; if ( isPowerOfTwo ( N ) ) { System . out . println ( " Yes " ) ; } else { System . out . println ( " No " ) ; } } }
class GFG {
static class Cantor { double start , end ; Cantor next ; } ; static Cantor Cantor ;
static Cantor startList ( Cantor head , double start_num , double end_num ) { if ( head == null ) { head = new Cantor ( ) ; head . start = start_num ; head . end = end_num ; head . next = null ; } return head ; }
static Cantor propagate ( Cantor head ) { Cantor temp = head ; if ( temp != null ) { Cantor newNode = new Cantor ( ) ; double diff = ( ( ( temp . end ) - ( temp . start ) ) / 3 ) ;
newNode . end = temp . end ; temp . end = ( ( temp . start ) + diff ) ; newNode . start = ( newNode . end ) - diff ;
newNode . next = temp . next ; temp . next = newNode ;
propagate ( temp . next . next ) ; } return head ; }
static void print ( Cantor temp ) { while ( temp != null ) { System . out . printf ( " [ % f ] ▁ - - ▁ [ % f ] " , temp . start , temp . end ) ; temp = temp . next ; } System . out . printf ( "NEW_LINE"); }
static void buildCantorSet ( int A , int B , int L ) { Cantor head = null ; head = startList ( head , A , B ) ; for ( int i = 0 ; i < L ; i ++ ) { System . out . printf ( " Level _ % d ▁ : ▁ " , i ) ; print ( head ) ; propagate ( head ) ; } System . out . printf ( " Level _ % d ▁ : ▁ " , L ) ; print ( head ) ; }
public static void main ( String [ ] args ) { int A = 0 ; int B = 9 ; int L = 2 ; buildCantorSet ( A , B , L ) ; } }
class GFG {
static void search ( String pat , String txt ) { int M = pat . length ( ) ; int N = txt . length ( ) ; int i = 0 ; while ( i <= N - M ) { int j ;
for ( j = 0 ; j < M ; j ++ ) if ( txt . charAt ( i + j ) != pat . charAt ( j ) ) break ;
{ System . out . println ( " Pattern ▁ found ▁ at ▁ index ▁ " + i ) ; i = i + M ; } else if ( j == 0 ) i = i + 1 ; else
i = i + j ; } }
public static void main ( String [ ] args ) { String txt = " ABCEABCDABCEABCD " ; String pat = " ABCD " ; search ( pat , txt ) ; } }
class GFG {
static void encrypt ( char input [ ] ) {
char evenPos = ' @ ' , oddPos = ' ! ' ; int repeat , ascii ; for ( int i = 0 ; i < input . length ; i ++ ) {
ascii = input [ i ] ; repeat = ascii >= 97 ? ascii - 96 : ascii - 64 ; for ( int j = 0 ; j < repeat ; j ++ ) {
if ( i % 2 == 0 ) System . out . printf ( " % c " , oddPos ) ; else System . out . printf ( " % c " , evenPos ) ; } } }
public static void main ( String [ ] args ) { char input [ ] = { ' A ' , ' b ' , ' C ' , ' d ' } ;
encrypt ( input ) ; } }
import java . io . * ; class GFG {
static boolean isPalRec ( String str , int s , int e ) {
if ( s == e ) return true ;
if ( ( str . charAt ( s ) ) != ( str . charAt ( e ) ) ) return false ;
if ( s < e + 1 ) return isPalRec ( str , s + 1 , e - 1 ) ; return true ; } static boolean isPalindrome ( String str ) { int n = str . length ( ) ;
if ( n == 0 ) return true ; return isPalRec ( str , 0 , n - 1 ) ; }
public static void main ( String args [ ] ) { String str = " geeg " ; if ( isPalindrome ( str ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
class GFG { static int myAtoi ( char [ ] str ) { int sign = 1 , base = 0 , i = 0 ;
while ( str [ i ] == ' ▁ ' ) { i ++ ; }
if ( str [ i ] == ' - ' str [ i ] == ' + ' ) { sign = 1 - 2 * ( str [ i ++ ] == ' - ' ? 1 : 0 ) ; }
while ( i < str . length && str [ i ] >= '0' && str [ i ] <= '9' ) {
if ( base > Integer . MAX_VALUE / 10 || ( base == Integer . MAX_VALUE / 10 && str [ i ] - '0' > 7 ) ) { if ( sign == 1 ) return Integer . MAX_VALUE ; else return Integer . MIN_VALUE ; } base = 10 * base + ( str [ i ++ ] - '0' ) ; } return base * sign ; }
public static void main ( String [ ] args ) { char str [ ] = " ▁ - 123" . toCharArray ( ) ;
int val = myAtoi ( str ) ; System . out . printf ( " % d ▁ " , val ) ; } }
import java . io . * ; class GFG {
static boolean fillUtil ( int res [ ] , int curr , int n ) {
if ( curr == 0 ) return true ;
int i ; for ( i = 0 ; i < 2 * n - curr - 1 ; i ++ ) {
if ( res [ i ] == 0 && res [ i + curr + 1 ] == 0 ) {
res [ i ] = res [ i + curr + 1 ] = curr ;
if ( fillUtil ( res , curr - 1 , n ) ) return true ;
res [ i ] = res [ i + curr + 1 ] = 0 ; } } return false ; }
static void fill ( int n ) {
int res [ ] = new int [ 2 * n ] ; int i ; for ( i = 0 ; i < 2 * n ; i ++ ) res [ i ] = 0 ;
if ( fillUtil ( res , n , n ) ) { for ( i = 0 ; i < 2 * n ; i ++ ) System . out . print ( res [ i ] + " ▁ " ) ; } else System . out . print ( " Not ▁ Possible " ) ; }
public static void main ( String [ ] args ) { fill ( 7 ) ; } }
import java . util . * ; class GFG {
static int findNumberOfDigits ( int n , int base ) {
int dig = ( ( int ) Math . floor ( Math . log ( n ) / Math . log ( base ) ) + 1 ) ;
return dig ; }
static boolean isAllKs ( int n , int b , int k ) { int len = findNumberOfDigits ( n , b ) ;
int sum = k * ( 1 - ( int ) Math . pow ( b , len ) ) / ( 1 - b ) ; return sum == n ; }
public static void main ( String [ ] args ) {
int N = 13 ;
int B = 3 ;
int K = 1 ;
if ( isAllKs ( N , B , K ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
import java . io . * ; class GFG {
static void CalPeri ( ) { int S = 5 , Perimeter ; Perimeter = 10 * S ; System . out . println ( " The ▁ Perimeter ▁ of ▁ Decagon ▁ is ▁ : ▁ " + Perimeter ) ; }
public static void main ( String [ ] args ) { CalPeri ( ) ; } }
import java . io . * ; import java . lang . Math ; class GFG {
static void distance ( float a1 , float b1 , float c1 , float a2 , float b2 , float c2 ) { float d = ( a1 * a2 + b1 * b2 + c1 * c2 ) ; float e1 = ( float ) Math . sqrt ( a1 * a1 + b1 * b1 + c1 * c1 ) ; float e2 = ( float ) Math . sqrt ( a2 * a2 + b2 * b2 + c2 * c2 ) ; d = d / ( e1 * e2 ) ; float pi = ( float ) 3.14159 ; float A = ( 180 / pi ) * ( float ) ( Math . acos ( d ) ) ; System . out . println ( " Angle ▁ is ▁ " + A + " ▁ degree " ) ; }
public static void main ( String [ ] args ) { float a1 = 1 ; float b1 = 1 ; float c1 = 2 ; float d1 = 1 ; float a2 = 2 ; float b2 = - 1 ; float c2 = 1 ; float d2 = - 4 ; distance ( a1 , b1 , c1 , a2 , b2 , c2 ) ; } }
import java . io . * ; class GFG {
static void mirror_point ( int a , int b , int c , int d , int x1 , int y1 , int z1 ) { float k = ( - a * x1 - b * y1 - c * z1 - d ) / ( float ) ( a * a + b * b + c * c ) ; float x2 = a * k + x1 ; float y2 = b * k + y1 ; float z2 = c * k + z1 ; float x3 = 2 * x2 - x1 ; float y3 = 2 * y2 - y1 ; float z3 = 2 * z2 - z1 ; System . out . print ( " x3 ▁ = ▁ " + x3 + " ▁ " ) ; System . out . print ( " y3 ▁ = ▁ " + y3 + " ▁ " ) ; System . out . print ( " z3 ▁ = ▁ " + z3 + " ▁ " ) ; }
public static void main ( String [ ] args ) { int a = 1 ; int b = - 2 ; int c = 0 ; int d = 0 ; int x1 = - 1 ; int y1 = 3 ; int z1 = 4 ;
mirror_point ( a , b , c , d , x1 , y1 , z1 ) ; } }
class GfG {
static class node { int data ; node left , right ; }
static int updatetree ( node root ) {
if ( root == null ) return 0 ; if ( root . left == null && root . right == null ) return root . data ;
int leftsum = updatetree ( root . left ) ; int rightsum = updatetree ( root . right ) ;
root . data += leftsum ;
return root . data + rightsum ; }
static void inorder ( node node ) { if ( node == null ) return ; inorder ( node . left ) ; System . out . print ( node . data + " ▁ " ) ; inorder ( node . right ) ; }
static node newNode ( int data ) { node node = new node ( ) ; node . data = data ; node . left = null ; node . right = null ; return ( node ) ; }
public static void main ( String [ ] args ) {
node root = newNode ( 1 ) ; root . left = newNode ( 2 ) ; root . right = newNode ( 3 ) ; root . left . left = newNode ( 4 ) ; root . left . right = newNode ( 5 ) ; root . right . right = newNode ( 6 ) ; updatetree ( root ) ; System . out . println ( " Inorder ▁ traversal ▁ of ▁ the ▁ modified ▁ tree ▁ is " ) ; inorder ( root ) ; } }
import java . util . Arrays ; class GFG {
static void calculateSpan ( int price [ ] , int n , int S [ ] ) {
S [ 0 ] = 1 ;
for ( int i = 1 ; i < n ; i ++ ) {
S [ i ] = 1 ;
for ( int j = i - 1 ; ( j >= 0 ) && ( price [ i ] >= price [ j ] ) ; j -- ) S [ i ] ++ ; } }
static void printArray ( int arr [ ] ) { System . out . print ( Arrays . toString ( arr ) ) ; }
public static void main ( String [ ] args ) { int price [ ] = { 10 , 4 , 5 , 90 , 120 , 80 } ; int n = price . length ; int S [ ] = new int [ n ] ;
calculateSpan ( price , n , S ) ;
printArray ( S ) ; } }
class Main {
static void printNGE ( int arr [ ] , int n ) { int next , i , j ; for ( i = 0 ; i < n ; i ++ ) { next = - 1 ; for ( j = i + 1 ; j < n ; j ++ ) { if ( arr [ i ] < arr [ j ] ) { next = arr [ j ] ; break ; } } System . out . println ( arr [ i ] + " ▁ - - ▁ " + next ) ; } }
public static void main ( String args [ ] ) { int arr [ ] = { 11 , 13 , 21 , 3 } ; int n = arr . length ; printNGE ( arr , n ) ; } }
class Node { int data ; Node left , right ; public Node ( int item ) { data = item ; left = right = null ; } }
class BinaryTree { Node root ; void mirror ( ) { root = mirror ( root ) ; } Node mirror ( Node node ) { if ( node == null ) return node ;
Node left = mirror ( node . left ) ; Node right = mirror ( node . right ) ;
node . left = right ; node . right = left ; return node ; } void inOrder ( ) { inOrder ( root ) ; }
void inOrder ( Node node ) { if ( node == null ) return ; inOrder ( node . left ) ; System . out . print ( node . data + " ▁ " ) ; inOrder ( node . right ) ; }
public static void main ( String args [ ] ) { BinaryTree tree = new BinaryTree ( ) ; tree . root = new Node ( 1 ) ; tree . root . left = new Node ( 2 ) ; tree . root . right = new Node ( 3 ) ; tree . root . left . left = new Node ( 4 ) ; tree . root . left . right = new Node ( 5 ) ;
System . out . println ( " Inorder ▁ traversal ▁ of ▁ input ▁ tree ▁ is ▁ : " ) ; tree . inOrder ( ) ; System . out . println ( " " ) ;
tree . mirror ( ) ;
System . out . println ( " Inorder ▁ traversal ▁ of ▁ binary ▁ tree ▁ is ▁ : ▁ " ) ; tree . inOrder ( ) ; } }
class Node { int data ; Node left , right ; Node ( int item ) { data = item ; left = right = null ; } } class BinaryTree { Node root ;
boolean IsFoldable ( Node node ) { if ( node == null ) return true ; return IsFoldableUtil ( node . left , node . right ) ; }
boolean IsFoldableUtil ( Node n1 , Node n2 ) {
if ( n1 == null && n2 == null ) return true ;
if ( n1 == null n2 == null ) return false ;
return IsFoldableUtil ( n1 . left , n2 . right ) && IsFoldableUtil ( n1 . right , n2 . left ) ; }
public static void main ( String args [ ] ) { BinaryTree tree = new BinaryTree ( ) ;
tree . root = new Node ( 1 ) ; tree . root . left = new Node ( 2 ) ; tree . root . right = new Node ( 3 ) ; tree . root . right . left = new Node ( 4 ) ; tree . root . left . right = new Node ( 5 ) ; if ( tree . IsFoldable ( tree . root ) ) System . out . println ( " tree ▁ is ▁ foldable " ) ; else System . out . println ( " Tree ▁ is ▁ not ▁ foldable " ) ; } }
class Node { int data ; Node left , right ; public Node ( int d ) { data = d ; left = right = null ; } } class BinaryTree { Node root ;
int isSumProperty ( Node node ) {
int left_data = 0 , right_data = 0 ;
if ( node == null || ( node . left == null && node . right == null ) ) return 1 ; else {
if ( node . left != null ) left_data = node . left . data ;
if ( node . right != null ) right_data = node . right . data ;
if ( ( node . data == left_data + right_data ) && ( isSumProperty ( node . left ) != 0 ) && isSumProperty ( node . right ) != 0 ) return 1 ; else return 0 ; } }
public static void main ( String [ ] args ) { BinaryTree tree = new BinaryTree ( ) ; tree . root = new Node ( 10 ) ; tree . root . left = new Node ( 8 ) ; tree . root . right = new Node ( 2 ) ; tree . root . left . left = new Node ( 3 ) ; tree . root . left . right = new Node ( 5 ) ; tree . root . right . right = new Node ( 2 ) ; if ( tree . isSumProperty ( tree . root ) != 0 ) System . out . println ( " The ▁ given ▁ tree ▁ satisfies ▁ children " + " ▁ sum ▁ property " ) ; else System . out . println ( " The ▁ given ▁ tree ▁ does ▁ not ▁ satisfy ▁ children " + " ▁ sum ▁ property " ) ; } }
class Test {
static int gcd ( int a , int b ) {
if ( a == 0 && b == 0 ) return 0 ; if ( a == 0 ) return b ; if ( b == 0 ) return a ;
if ( a == b ) return a ;
if ( a > b ) return gcd ( a - b , b ) ; return gcd ( a , b - a ) ; }
public static void main ( String [ ] args ) { int a = 98 , b = 56 ; System . out . println ( " GCD ▁ of ▁ " + a + " ▁ and ▁ " + b + " ▁ is ▁ " + gcd ( a , b ) ) ; } }
public class GFG {
static int msbPos ( int n ) { int pos = 0 ; while ( n != 0 ) { pos ++ ;
n = n >> 1 ; } return pos ; }
static int josephify ( int n ) {
int position = msbPos ( n ) ;
int j = 1 << ( position - 1 ) ;
n = n ^ j ;
n = n << 1 ;
n = n | 1 ; return n ; }
public static void main ( String [ ] args ) { int n = 41 ; System . out . println ( josephify ( n ) ) ; } }
import java . io . * ; class GFG {
static int pairAndSum ( int arr [ ] , int n ) {
for ( int i = 0 ; i < 32 ; i ++ ) {
int k = 0 ; for ( int j = 0 ; j < n ; j ++ ) { if ( ( arr [ j ] & ( 1 << i ) ) != 0 ) k ++ ; }
ans += ( 1 << i ) * ( k * ( k - 1 ) / 2 ) ; } return ans ; }
public static void main ( String args [ ] ) { int arr [ ] = { 5 , 10 , 15 } ; int n = arr . length ; System . out . println ( pairAndSum ( arr , n ) ) ; } }
static int countSquares ( int n ) {
return ( n * ( n + 1 ) / 2 ) * ( 2 * n + 1 ) / 3 ; }
public static void main ( String [ ] args ) { int n = 3 ; System . out . println ( " Count ▁ of ▁ squares ▁ is ▁ " + countSquares ( n ) ) ; } }
class Test {
static int gcd ( int a , int b ) {
if ( a == 0 ) return b ; if ( b == 0 ) return a ;
if ( a == b ) return a ;
if ( a > b ) return gcd ( a - b , b ) ; return gcd ( a , b - a ) ; }
public static void main ( String [ ] args ) { int a = 98 , b = 56 ; System . out . println ( " GCD ▁ of ▁ " + a + " ▁ and ▁ " + b + " ▁ is ▁ " + gcd ( a , b ) ) ; } }
class GFG {
static int maxsize = 100005 ;
static int [ ] xor_tree = new int [ maxsize ] ;
static void construct_Xor_Tree_Util ( int [ ] current , int start , int end , int x ) {
if ( start == end ) { xor_tree [ x ] = current [ start ] ;
return ; }
int left = x * 2 + 1 ;
int right = x * 2 + 2 ;
int mid = start + ( end - start ) / 2 ;
construct_Xor_Tree_Util ( current , start , mid , left ) ; construct_Xor_Tree_Util ( current , mid + 1 , end , right ) ;
xor_tree [ x ] = ( xor_tree [ left ] ^ xor_tree [ right ] ) ; }
static void construct_Xor_Tree ( int [ ] arr , int n ) { construct_Xor_Tree_Util ( arr , 0 , n - 1 , 0 ) ; }
public static void main ( String [ ] args ) {
int [ ] leaf_nodes = { 40 , 32 , 12 , 1 , 4 , 3 , 2 , 7 } ; int n = leaf_nodes . length ;
construct_Xor_Tree ( leaf_nodes , n ) ;
int x = ( int ) ( Math . ceil ( Math . log ( n ) ) ) ;
int max_size = 2 * ( int ) Math . pow ( 2 , x ) - 1 ; System . out . print ( "Nodes of the XOR Tree:NEW_LINE"); for ( int i = 0 ; i < max_size ; i ++ ) { System . out . print ( xor_tree [ i ] + " ▁ " ) ; }
int root = 0 ;
System . out . print ( " Root : " + xor_tree[root]); } }
import java . util . * ; class Main { public static int swapBits ( int n , int p1 , int p2 ) {
n ^= 1 << p1 ; n ^= 1 << p2 ; return n ; }
public static void main ( String [ ] args ) { System . out . print ( " Result ▁ = ▁ " + swapBits ( 28 , 0 , 3 ) ) ; } }
class Node { int data ; Node left , right ; Node ( int item ) { data = item ; left = right = null ; } } class BinaryTree { Node root ;
boolean isSibling ( Node node , Node a , Node b ) {
if ( node == null ) return false ; return ( ( node . left == a && node . right == b ) || ( node . left == b && node . right == a ) || isSibling ( node . left , a , b ) || isSibling ( node . right , a , b ) ) ; }
int level ( Node node , Node ptr , int lev ) {
if ( node == null ) return 0 ; if ( node == ptr ) return lev ;
int l = level ( node . left , ptr , lev + 1 ) ; if ( l != 0 ) return l ;
return level ( node . right , ptr , lev + 1 ) ; }
boolean isCousin ( Node node , Node a , Node b ) {
return ( ( level ( node , a , 1 ) == level ( node , b , 1 ) ) && ( ! isSibling ( node , a , b ) ) ) ; }
public static void main ( String args [ ] ) { BinaryTree tree = new BinaryTree ( ) ; tree . root = new Node ( 1 ) ; tree . root . left = new Node ( 2 ) ; tree . root . right = new Node ( 3 ) ; tree . root . left . left = new Node ( 4 ) ; tree . root . left . right = new Node ( 5 ) ; tree . root . left . right . right = new Node ( 15 ) ; tree . root . right . left = new Node ( 6 ) ; tree . root . right . right = new Node ( 7 ) ; tree . root . right . left . right = new Node ( 8 ) ; Node Node1 , Node2 ; Node1 = tree . root . left . left ; Node2 = tree . root . right . right ; if ( tree . isCousin ( tree . root , Node1 , Node2 ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
class Node { int data ; Node left , right ; Node ( int item ) { data = item ; left = right = null ; } } class Leaf { int leaflevel = 0 ; } class BinaryTree { Node root ; Leaf mylevel = new Leaf ( ) ;
boolean checkUtil ( Node node , int level , Leaf leafLevel ) {
if ( node == null ) return true ;
if ( node . left == null && node . right == null ) {
if ( leafLevel . leaflevel == 0 ) {
leafLevel . leaflevel = level ; return true ; }
return ( level == leafLevel . leaflevel ) ; }
return checkUtil ( node . left , level + 1 , leafLevel ) && checkUtil ( node . right , level + 1 , leafLevel ) ; }
boolean check ( Node node ) { int level = 0 ; return checkUtil ( node , level , mylevel ) ; }
public static void main ( String args [ ] ) {
BinaryTree tree = new BinaryTree ( ) ; tree . root = new Node ( 12 ) ; tree . root . left = new Node ( 5 ) ; tree . root . left . left = new Node ( 3 ) ; tree . root . left . right = new Node ( 9 ) ; tree . root . left . left . left = new Node ( 1 ) ; tree . root . left . right . left = new Node ( 1 ) ; if ( tree . check ( tree . root ) ) System . out . println ( " Leaves ▁ are ▁ at ▁ same ▁ level " ) ; else System . out . println ( " Leaves ▁ are ▁ not ▁ at ▁ same ▁ level " ) ; } }
class Node { int data ; Node left , right ; Node ( int item ) { data = item ; left = right = null ; } } class BinaryTree { Node root ;
boolean isFullTree ( Node node ) {
if ( node == null ) return true ;
if ( node . left == null && node . right == null ) return true ;
if ( ( node . left != null ) && ( node . right != null ) ) return ( isFullTree ( node . left ) && isFullTree ( node . right ) ) ;
return false ; }
public static void main ( String args [ ] ) { BinaryTree tree = new BinaryTree ( ) ; tree . root = new Node ( 10 ) ; tree . root . left = new Node ( 20 ) ; tree . root . right = new Node ( 30 ) ; tree . root . left . right = new Node ( 40 ) ; tree . root . left . left = new Node ( 50 ) ; tree . root . right . left = new Node ( 60 ) ; tree . root . left . left . left = new Node ( 80 ) ; tree . root . right . right = new Node ( 70 ) ; tree . root . left . left . right = new Node ( 90 ) ; tree . root . left . right . left = new Node ( 80 ) ; tree . root . left . right . right = new Node ( 90 ) ; tree . root . right . left . left = new Node ( 80 ) ; tree . root . right . left . right = new Node ( 90 ) ; tree . root . right . right . left = new Node ( 80 ) ; tree . root . right . right . right = new Node ( 90 ) ; if ( tree . isFullTree ( tree . root ) ) System . out . print ( " The ▁ binary ▁ tree ▁ is ▁ full " ) ; else System . out . print ( " The ▁ binary ▁ tree ▁ is ▁ not ▁ full " ) ; } }
import java . io . * ; class GFG {
static void printAlter ( int [ ] arr , int N ) {
for ( int currIndex = 0 ; currIndex < N ; currIndex += 2 ) {
System . out . print ( arr [ currIndex ] + " ▁ " ) ; } }
public static void main ( String [ ] args ) { int [ ] arr = { 1 , 2 , 3 , 4 , 5 } ; int N = arr . length ; printAlter ( arr , N ) ; } }
class Node { int data ; Node left , right ; Node ( int item ) { data = item ; left = right = null ; } } class BinaryTree { Node root1 , root2 ;
boolean identicalTrees ( Node a , Node b ) {
if ( a == null && b == null ) return true ;
if ( a != null && b != null ) return ( a . data == b . data && identicalTrees ( a . left , b . left ) && identicalTrees ( a . right , b . right ) ) ;
return false ; }
public static void main ( String [ ] args ) { BinaryTree tree = new BinaryTree ( ) ; tree . root1 = new Node ( 1 ) ; tree . root1 . left = new Node ( 2 ) ; tree . root1 . right = new Node ( 3 ) ; tree . root1 . left . left = new Node ( 4 ) ; tree . root1 . left . right = new Node ( 5 ) ; tree . root2 = new Node ( 1 ) ; tree . root2 . left = new Node ( 2 ) ; tree . root2 . right = new Node ( 3 ) ; tree . root2 . left . left = new Node ( 4 ) ; tree . root2 . left . right = new Node ( 5 ) ; if ( tree . identicalTrees ( tree . root1 , tree . root2 ) ) System . out . println ( " Both ▁ trees ▁ are ▁ identical " ) ; else System . out . println ( " Trees ▁ are ▁ not ▁ identical " ) ; } }
class GfG {
static class Node { int data ; Node left , right ; }
static Node newNode ( int item ) { Node temp = new Node ( ) ; temp . data = item ; temp . left = null ; temp . right = null ; return temp ; }
static int getLevel ( Node root , Node node , int level ) {
if ( root == null ) return 0 ; if ( root == node ) return level ;
int downlevel = getLevel ( root . left , node , level + 1 ) ; if ( downlevel != 0 ) return downlevel ;
return getLevel ( root . right , node , level + 1 ) ; }
static void printGivenLevel ( Node root , Node node , int level ) {
if ( root == null level < 2 ) return ;
if ( level == 2 ) { if ( root . left == node root . right == node ) return ; if ( root . left != null ) System . out . print ( root . left . data + " ▁ " ) ; if ( root . right != null ) System . out . print ( root . right . data + " ▁ " ) ; }
else if ( level > 2 ) { printGivenLevel ( root . left , node , level - 1 ) ; printGivenLevel ( root . right , node , level - 1 ) ; } }
static void printCousins ( Node root , Node node ) {
int level = getLevel ( root , node , 1 ) ;
printGivenLevel ( root , node , level ) ; }
public static void main ( String [ ] args ) { Node root = newNode ( 1 ) ; root . left = newNode ( 2 ) ; root . right = newNode ( 3 ) ; root . left . left = newNode ( 4 ) ; root . left . right = newNode ( 5 ) ; root . left . right . right = newNode ( 15 ) ; root . right . left = newNode ( 6 ) ; root . right . right = newNode ( 7 ) ; root . right . left . right = newNode ( 8 ) ; printCousins ( root , root . left . right ) ; } }
class Node { int data ; Node left , right ; Node ( int item ) { data = item ; left = right = null ; } } class BinaryTree { Node root ;
void printPaths ( Node node ) { int path [ ] = new int [ 1000 ] ; printPathsRecur ( node , path , 0 ) ; }
void printPathsRecur ( Node node , int path [ ] , int pathLen ) { if ( node == null ) return ;
path [ pathLen ] = node . data ; pathLen ++ ;
if ( node . left == null && node . right == null ) printArray ( path , pathLen ) ; else {
printPathsRecur ( node . left , path , pathLen ) ; printPathsRecur ( node . right , path , pathLen ) ; } }
void printArray ( int ints [ ] , int len ) { int i ; for ( i = 0 ; i < len ; i ++ ) System . out . print ( ints [ i ] + " ▁ " ) ; System . out . println ( " " ) ; }
public static void main ( String [ ] args ) { BinaryTree tree = new BinaryTree ( ) ; tree . root = new Node ( 1 ) ; tree . root . left = new Node ( 2 ) ; tree . root . right = new Node ( 3 ) ; tree . root . left . left = new Node ( 4 ) ; tree . root . left . right = new Node ( 5 ) ;
tree . printPaths ( tree . root ) ; } }
import java . util . * ; class GFG {
public static void leftRotate ( int arr [ ] , int d , int n ) { leftRotateRec ( arr , 0 , d , n ) ; } public static void leftRotateRec ( int arr [ ] , int i , int d , int n ) {
if ( d == 0 d == n ) return ;
if ( n - d == d ) { swap ( arr , i , n - d + i , d ) ; return ; }
if ( d < n - d ) { swap ( arr , i , n - d + i , d ) ; leftRotateRec ( arr , i , d , n - d ) ; }
else { swap ( arr , i , d , n - d ) ; leftRotateRec ( arr , n - d + i , 2 * d - n , d ) ; } }
public static void printArray ( int arr [ ] , int size ) { int i ; for ( i = 0 ; i < size ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; System . out . println ( ) ; }
public static void swap ( int arr [ ] , int fi , int si , int d ) { int i , temp ; for ( i = 0 ; i < d ; i ++ ) { temp = arr [ fi + i ] ; arr [ fi + i ] = arr [ si + i ] ; arr [ si + i ] = temp ; } }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 , 6 , 7 } ; leftRotate ( arr , 2 , 7 ) ; printArray ( arr , 7 ) ; } }
static void leftRotate ( int arr [ ] , int d , int n ) { int i , j ; if ( d == 0 d == n ) return ; i = d ; j = n - d ; while ( i != j ) {
if ( i < j ) { swap ( arr , d - i , d + j - i , i ) ; j -= i ; }
else { swap ( arr , d - i , d , j ) ; i -= j ; } }
swap ( arr , d - i , d , i ) ; }
import java . util . Arrays ; public class Test { static int arr [ ] = new int [ ] { 1 , 2 , 3 , 4 , 5 } ;
static void rotate ( ) { int i = 0 , j = arr . length - 1 ; while ( i != j ) { int temp = arr [ i ] ; arr [ i ] = arr [ j ] ; arr [ j ] = temp ; i ++ ; } }
public static void main ( String [ ] args ) { System . out . println ( " Given ▁ Array ▁ is " ) ; System . out . println ( Arrays . toString ( arr ) ) ; rotate ( ) ; System . out . println ( " Rotated ▁ Array ▁ is " ) ; System . out . println ( Arrays . toString ( arr ) ) ; } }
import java . util . * ; import java . io . * ; class Main {
static void selectionSort ( String arr [ ] , int n ) {
for ( int i = 0 ; i < n - 1 ; i ++ ) {
int min_index = i ; String minStr = arr [ i ] ; for ( int j = i + 1 ; j < n ; j ++ ) {
if ( arr [ j ] . compareTo ( minStr ) < 0 ) {
minStr = arr [ j ] ; min_index = j ; } }
if ( min_index != i ) { String temp = arr [ min_index ] ; arr [ min_index ] = arr [ i ] ; arr [ i ] = temp ; } } }
public static void main ( String args [ ] ) { String arr [ ] = { " GeeksforGeeks " , " Practice . GeeksforGeeks " , " GeeksQuiz " } ; int n = arr . length ; System . out . println ( " Given ▁ array ▁ is " ) ;
for ( int i = 0 ; i < n ; i ++ ) { System . out . println ( i + " : ▁ " + arr [ i ] ) ; } System . out . println ( ) ; selectionSort ( arr , n ) ; System . out . println ( " Sorted ▁ array ▁ is " ) ;
for ( int i = 0 ; i < n ; i ++ ) { System . out . println ( i + " : ▁ " + arr [ i ] ) ; } } }
class RearrangeArray {
void rearrangeNaive ( int arr [ ] , int n ) {
int temp [ ] = new int [ n ] ; int i ;
for ( i = 0 ; i < n ; i ++ ) temp [ arr [ i ] ] = i ;
for ( i = 0 ; i < n ; i ++ ) arr [ i ] = temp [ i ] ; }
void printArray ( int arr [ ] , int n ) { int i ; for ( i = 0 ; i < n ; i ++ ) { System . out . print ( arr [ i ] + " ▁ " ) ; } System . out . println ( " " ) ; }
public static void main ( String [ ] args ) { RearrangeArray arrange = new RearrangeArray ( ) ; int arr [ ] = { 1 , 3 , 0 , 2 } ; int n = arr . length ; System . out . println ( " Given ▁ array ▁ is ▁ " ) ; arrange . printArray ( arr , n ) ; arrange . rearrangeNaive ( arr , n ) ; System . out . println ( " Modified ▁ array ▁ is ▁ " ) ; arrange . printArray ( arr , n ) ; } }
class Test { static int arr [ ] = { 10 , 324 , 45 , 90 , 9808 } ;
static int largest ( ) { int i ;
int max = arr [ 0 ] ;
for ( i = 1 ; i < arr . length ; i ++ ) if ( arr [ i ] > max ) max = arr [ i ] ; return max ; }
public static void main ( String [ ] args ) { System . out . println ( " Largest ▁ in ▁ given ▁ array ▁ is ▁ " + largest ( ) ) ; } }
class GFG {
public static void print2largest ( int arr [ ] , int arr_size ) { int i , first , second ;
if ( arr_size < 2 ) { System . out . print ( " ▁ Invalid ▁ Input ▁ " ) ; return ; } first = second = Integer . MIN_VALUE ; for ( i = 0 ; i < arr_size ; i ++ ) {
if ( arr [ i ] > first ) { second = first ; first = arr [ i ] ; }
else if ( arr [ i ] > second && arr [ i ] != first ) second = arr [ i ] ; } if ( second == Integer . MIN_VALUE ) System . out . print ( " There ▁ is ▁ no ▁ second ▁ largest " + " elementNEW_LINE"); else System . out . print ( " The ▁ second ▁ largest ▁ element " + " ▁ is ▁ " + second ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 12 , 35 , 1 , 10 , 34 , 1 } ; int n = arr . length ; print2largest ( arr , n ) ; } }
static class Pair { int min ; int max ; } static Pair getMinMax ( int arr [ ] , int n ) { Pair minmax = new Pair ( ) ; int i ;
if ( n == 1 ) { minmax . max = arr [ 0 ] ; minmax . min = arr [ 0 ] ; return minmax ; }
if ( arr [ 0 ] > arr [ 1 ] ) { minmax . max = arr [ 0 ] ; minmax . min = arr [ 1 ] ; } else { minmax . max = arr [ 1 ] ; minmax . min = arr [ 0 ] ; } for ( i = 2 ; i < n ; i ++ ) { if ( arr [ i ] > minmax . max ) { minmax . max = arr [ i ] ; } else if ( arr [ i ] < minmax . min ) { minmax . min = arr [ i ] ; } } return minmax ; }
public static void main ( String args [ ] ) { int arr [ ] = { 1000 , 11 , 445 , 1 , 330 , 3000 } ; int arr_size = 6 ; Pair minmax = getMinMax ( arr , arr_size ) ; System . out . printf ( " Minimum element is % d ", minmax.min); System . out . printf ( " Maximum element is % d ", minmax.max); } }
public class GFG {
static class Pair { int min ; int max ; } static Pair getMinMax ( int arr [ ] , int n ) { Pair minmax = new Pair ( ) ; int i ;
if ( n % 2 == 0 ) { if ( arr [ 0 ] > arr [ 1 ] ) { minmax . max = arr [ 0 ] ; minmax . min = arr [ 1 ] ; } else { minmax . min = arr [ 0 ] ; minmax . max = arr [ 1 ] ; }
i = 2 ; }
else { minmax . min = arr [ 0 ] ; minmax . max = arr [ 0 ] ;
i = 1 ; }
while ( i < n - 1 ) { if ( arr [ i ] > arr [ i + 1 ] ) { if ( arr [ i ] > minmax . max ) { minmax . max = arr [ i ] ; } if ( arr [ i + 1 ] < minmax . min ) { minmax . min = arr [ i + 1 ] ; } } else { if ( arr [ i + 1 ] > minmax . max ) { minmax . max = arr [ i + 1 ] ; } if ( arr [ i ] < minmax . min ) { minmax . min = arr [ i ] ; } }
i += 2 ; } return minmax ; }
public static void main ( String args [ ] ) { int arr [ ] = { 1000 , 11 , 445 , 1 , 330 , 3000 } ; int arr_size = 6 ; Pair minmax = getMinMax ( arr , arr_size ) ; System . out . printf ( " Minimum element is % d ", minmax.min); System . out . printf ( " Maximum element is % d ", minmax.max); } }
import java . util . * ; import java . io . * ; class GFG {
static int minJumps ( int arr [ ] , int l , int h ) {
if ( h == l ) return 0 ;
if ( arr [ l ] == 0 ) return Integer . MAX_VALUE ;
int min = Integer . MAX_VALUE ; for ( int i = l + 1 ; i <= h && i <= l + arr [ l ] ; i ++ ) { int jumps = minJumps ( arr , i , h ) ; if ( jumps != Integer . MAX_VALUE && jumps + 1 < min ) min = jumps + 1 ; } return min ; }
public static void main ( String args [ ] ) { int arr [ ] = { 1 , 3 , 6 , 3 , 2 , 3 , 6 , 8 , 9 , 5 } ; int n = arr . length ; System . out . print ( " Minimum ▁ number ▁ of ▁ jumps ▁ to ▁ reach ▁ end ▁ is ▁ " + minJumps ( arr , 0 , n - 1 ) ) ; } }
class SmallestSubArraySum {
static int smallestSubWithSum ( int arr [ ] , int n , int x ) {
int min_len = n + 1 ;
for ( int start = 0 ; start < n ; start ++ ) {
int curr_sum = arr [ start ] ;
if ( curr_sum > x ) return 1 ;
for ( int end = start + 1 ; end < n ; end ++ ) {
curr_sum += arr [ end ] ;
if ( curr_sum > x && ( end - start + 1 ) < min_len ) min_len = ( end - start + 1 ) ; } } return min_len ; }
public static void main ( String [ ] args ) { int arr1 [ ] = { 1 , 4 , 45 , 6 , 10 , 19 } ; int x = 51 ; int n1 = arr1 . length ; int res1 = smallestSubWithSum ( arr1 , n1 , x ) ; if ( res1 == n1 + 1 ) System . out . println ( " Not ▁ Possible " ) ; else System . out . println ( res1 ) ; int arr2 [ ] = { 1 , 10 , 5 , 2 , 7 } ; int n2 = arr2 . length ; x = 9 ; int res2 = smallestSubWithSum ( arr2 , n2 , x ) ; if ( res2 == n2 + 1 ) System . out . println ( " Not ▁ Possible " ) ; else System . out . println ( res2 ) ; int arr3 [ ] = { 1 , 11 , 100 , 1 , 0 , 200 , 3 , 2 , 1 , 250 } ; int n3 = arr3 . length ; x = 280 ; int res3 = smallestSubWithSum ( arr3 , n3 , x ) ; if ( res3 == n3 + 1 ) System . out . println ( " Not ▁ Possible " ) ; else System . out . println ( res3 ) ; } }
class Node { int key ; Node left , right ; public Node ( int item ) { key = item ; left = right = null ; } } class BinaryTree {
void printPostorder ( Node node ) { if ( node == null ) return ;
printPostorder ( node . left ) ;
printPostorder ( node . right ) ;
System . out . print ( node . key + " ▁ " ) ; }
void printInorder ( Node node ) { if ( node == null ) return ;
printInorder ( node . left ) ;
System . out . print ( node . key + " ▁ " ) ;
printInorder ( node . right ) ; }
void printPreorder ( Node node ) { if ( node == null ) return ;
System . out . print ( node . key + " ▁ " ) ;
printPreorder ( node . left ) ;
printPreorder ( node . right ) ; }
public static void main ( String [ ] args ) { BinaryTree tree = new BinaryTree ( ) ; tree . root = new Node ( 1 ) ; tree . root . left = new Node ( 2 ) ; tree . root . right = new Node ( 3 ) ; tree . root . left . left = new Node ( 4 ) ; tree . root . left . right = new Node ( 5 ) ; System . out . println ( " Preorder ▁ traversal ▁ of ▁ binary ▁ tree ▁ is ▁ " ) ; tree . printPreorder ( ) ; System . out . println ( " Inorder traversal of binary tree is "); tree . printInorder ( ) ; System . out . println ( " Postorder traversal of binary tree is "); tree . printPostorder ( ) ; } }
class Node { int data ; Node left ; Node right ; public Node ( int data ) { this . data = data ; left = null ; right = null ; } }
public void print ( Node root ) { if ( root == null ) return ; print ( root . left ) ; System . out . print ( root . data + " ▁ " ) ; print ( root . right ) ; } }
class BinaryTree { Node root ; public Node prune ( Node root , int sum ) {
if ( root == null ) return null ;
root . left = prune ( root . left , sum - root . data ) ; root . right = prune ( root . right , sum - root . data ) ;
if ( isLeaf ( root ) ) { if ( sum > root . data ) root = null ; } return root ; }
public class GFG { public static void main ( String args [ ] ) { BinaryTree tree = new BinaryTree ( ) ; tree . root = new Node ( 1 ) ; tree . root . left = new Node ( 2 ) ; tree . root . right = new Node ( 3 ) ; tree . root . left . left = new Node ( 4 ) ; tree . root . left . right = new Node ( 5 ) ; tree . root . right . left = new Node ( 6 ) ; tree . root . right . right = new Node ( 7 ) ; tree . root . left . left . left = new Node ( 8 ) ; tree . root . left . left . right = new Node ( 9 ) ; tree . root . left . right . left = new Node ( 12 ) ; tree . root . right . right . left = new Node ( 10 ) ; tree . root . right . right . left . right = new Node ( 11 ) ; tree . root . left . left . right . left = new Node ( 13 ) ; tree . root . left . left . right . right = new Node ( 14 ) ; tree . root . left . left . right . right . left = new Node ( 15 ) ; System . out . println ( " Tree ▁ before ▁ truncation " ) ; tree . print ( tree . root ) ; tree . prune ( tree . root , 45 ) ; System . out . println ( " Tree after truncation "); tree . print ( tree . root ) ; } }
class MergeArrays {
void moveToEnd ( int mPlusN [ ] , int size ) { int i , j = size - 1 ; for ( i = size - 1 ; i >= 0 ; i -- ) { if ( mPlusN [ i ] != - 1 ) { mPlusN [ j ] = mPlusN [ i ] ; j -- ; } } }
void merge ( int mPlusN [ ] , int N [ ] , int m , int n ) { int i = n ;
int j = 0 ;
int k = 0 ;
while ( k < ( m + n ) ) {
if ( ( i < ( m + n ) && mPlusN [ i ] <= N [ j ] ) || ( j == n ) ) { mPlusN [ k ] = mPlusN [ i ] ; k ++ ; i ++ ; }
else { mPlusN [ k ] = N [ j ] ; k ++ ; j ++ ; } } }
void printArray ( int arr [ ] , int size ) { int i ; for ( i = 0 ; i < size ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; System . out . println ( " " ) ; }
public static void main ( String [ ] args ) { MergeArrays mergearray = new MergeArrays ( ) ;
int mPlusN [ ] = { 2 , 8 , - 1 , - 1 , - 1 , 13 , - 1 , 15 , 20 } ; int N [ ] = { 5 , 7 , 9 , 25 } ; int n = N . length ; int m = mPlusN . length - n ;
mergearray . moveToEnd ( mPlusN , m + n ) ;
mergearray . merge ( mPlusN , N , m , n ) ;
mergearray . printArray ( mPlusN , m + n ) ; } }
import java . util . * ; class GFG {
public static long getCount ( int n , int k ) {
if ( n == 1 ) return 10 ;
long dp [ ] = new long [ 11 ] ;
long next [ ] = new long [ 11 ] ;
for ( int i = 1 ; i <= 9 ; i ++ ) dp [ i ] = 1 ;
for ( int i = 2 ; i <= n ; i ++ ) { for ( int j = 0 ; j <= 9 ; j ++ ) {
int l = Math . max ( 0 , j - k ) ; int r = Math . min ( 9 , j + k ) ;
next [ l ] += dp [ j ] ; next [ r + 1 ] -= dp [ j ] ; }
for ( int j = 1 ; j <= 9 ; j ++ ) next [ j ] += next [ j - 1 ] ;
for ( int j = 0 ; j < 10 ; j ++ ) { dp [ j ] = next [ j ] ; next [ j ] = 0 ; } }
long count = 0 ; for ( int i = 0 ; i <= 9 ; i ++ ) count += dp [ i ] ;
return count ; }
public static void main ( String [ ] args ) { int n = 2 , k = 1 ; System . out . println ( getCount ( n , k ) ) ; } }
class Test { static int arr [ ] = new int [ ] { 1 , 20 , 6 , 4 , 5 } ; static int getInvCount ( int n ) { int inv_count = 0 ; for ( int i = 0 ; i < n - 1 ; i ++ ) for ( int j = i + 1 ; j < n ; j ++ ) if ( arr [ i ] > arr [ j ] ) inv_count ++ ; return inv_count ; }
public static void main ( String [ ] args ) { System . out . println ( " Number ▁ of ▁ inversions ▁ are ▁ " + getInvCount ( arr . length ) ) ; } }
import java . util . * ; import java . lang . * ; class Main { static void minAbsSumPair ( int arr [ ] , int arr_size ) { int inv_count = 0 ; int l , r , min_sum , sum , min_l , min_r ;
if ( arr_size < 2 ) { System . out . println ( " Invalid ▁ Input " ) ; return ; }
min_l = 0 ; min_r = 1 ; min_sum = arr [ 0 ] + arr [ 1 ] ; for ( l = 0 ; l < arr_size - 1 ; l ++ ) { for ( r = l + 1 ; r < arr_size ; r ++ ) { sum = arr [ l ] + arr [ r ] ; if ( Math . abs ( min_sum ) > Math . abs ( sum ) ) { min_sum = sum ; min_l = l ; min_r = r ; } } } System . out . println ( " ▁ The ▁ two ▁ elements ▁ whose ▁ " + " sum ▁ is ▁ minimum ▁ are ▁ " + arr [ min_l ] + " ▁ and ▁ " + arr [ min_r ] ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 60 , - 10 , 70 , - 80 , 85 } ; minAbsSumPair ( arr , 6 ) ; } }
class FindUnion {
static int printUnion ( int arr1 [ ] , int arr2 [ ] , int m , int n ) { int i = 0 , j = 0 ; while ( i < m && j < n ) { if ( arr1 [ i ] < arr2 [ j ] ) System . out . print ( arr1 [ i ++ ] + " ▁ " ) ; else if ( arr2 [ j ] < arr1 [ i ] ) System . out . print ( arr2 [ j ++ ] + " ▁ " ) ; else { System . out . print ( arr2 [ j ++ ] + " ▁ " ) ; i ++ ; } }
while ( i < m ) System . out . print ( arr1 [ i ++ ] + " ▁ " ) ; while ( j < n ) System . out . print ( arr2 [ j ++ ] + " ▁ " ) ; return 0 ; }
public static void main ( String args [ ] ) { int arr1 [ ] = { 1 , 2 , 4 , 5 , 6 } ; int arr2 [ ] = { 2 , 3 , 5 , 7 } ; int m = arr1 . length ; int n = arr2 . length ; printUnion ( arr1 , arr2 , m , n ) ; } }
class FindIntersection {
static void printIntersection ( int arr1 [ ] , int arr2 [ ] , int m , int n ) { int i = 0 , j = 0 ; while ( i < m && j < n ) { if ( arr1 [ i ] < arr2 [ j ] ) i ++ ; else if ( arr2 [ j ] < arr1 [ i ] ) j ++ ; else { System . out . print ( arr2 [ j ++ ] + " ▁ " ) ; i ++ ; } } }
public static void main ( String args [ ] ) { int arr1 [ ] = { 1 , 2 , 4 , 5 , 6 } ; int arr2 [ ] = { 2 , 3 , 5 , 7 } ; int m = arr1 . length ; int n = arr2 . length ;
printIntersection ( arr1 , arr2 , m , n ) ; } }
class Node { int data ; Node left , right ; Node ( int item ) { data = item ; left = right = null ; } }
boolean printPath ( Node node , Node target_leaf ) {
if ( node == null ) return false ;
if ( node == target_leaf || printPath ( node . left , target_leaf ) || printPath ( node . right , target_leaf ) ) { System . out . print ( node . data + " ▁ " ) ; return true ; } return false ; }
void getTargetLeaf ( Node node , Maximum max_sum_ref , int curr_sum ) { if ( node == null ) return ;
curr_sum = curr_sum + node . data ;
if ( node . left == null && node . right == null ) { if ( curr_sum > max_sum_ref . max_no ) { max_sum_ref . max_no = curr_sum ; target_leaf = node ; } }
getTargetLeaf ( node . left , max_sum_ref , curr_sum ) ; getTargetLeaf ( node . right , max_sum_ref , curr_sum ) ; }
int maxSumPath ( ) {
if ( root == null ) return 0 ;
getTargetLeaf ( root , max , 0 ) ;
printPath ( root , target_leaf ) ;
return max . max_no ; }
public static void main ( String args [ ] ) { BinaryTree tree = new BinaryTree ( ) ; tree . root = new Node ( 10 ) ; tree . root . left = new Node ( - 2 ) ; tree . root . right = new Node ( 7 ) ; tree . root . left . left = new Node ( 8 ) ; tree . root . left . right = new Node ( - 4 ) ; System . out . println ( " Following ▁ are ▁ the ▁ nodes ▁ " + " on ▁ maximum ▁ sum ▁ path " ) ; int sum = tree . maxSumPath ( ) ; System . out . println ( " " ) ; System . out . println ( " Sum ▁ of ▁ nodes ▁ is ▁ : ▁ " + sum ) ; } }
import java . io . * ; class countzot {
static void sort012 ( int a [ ] , int arr_size ) { int lo = 0 ; int hi = arr_size - 1 ; int mid = 0 , temp = 0 ; while ( mid <= hi ) { switch ( a [ mid ] ) { case 0 : { temp = a [ lo ] ; a [ lo ] = a [ mid ] ; a [ mid ] = temp ; lo ++ ; mid ++ ; break ; } case 1 : mid ++ ; break ; case 2 : { temp = a [ mid ] ; a [ mid ] = a [ hi ] ; a [ hi ] = temp ; hi -- ; break ; } } } }
static void printArray ( int arr [ ] , int arr_size ) { int i ; for ( i = 0 ; i < arr_size ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; System . out . println ( " " ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 0 , 1 , 1 , 0 , 1 , 2 , 1 , 2 , 0 , 0 , 0 , 1 } ; int arr_size = arr . length ; sort012 ( arr , arr_size ) ; System . out . println ( " Array ▁ after ▁ seggregation ▁ " ) ; printArray ( arr , arr_size ) ; } }
class Main { static void printUnsorted ( int arr [ ] , int n ) { int s = 0 , e = n - 1 , i , max , min ;
for ( s = 0 ; s < n - 1 ; s ++ ) { if ( arr [ s ] > arr [ s + 1 ] ) break ; } if ( s == n - 1 ) { System . out . println ( " The ▁ complete ▁ array ▁ is ▁ sorted " ) ; return ; }
for ( e = n - 1 ; e > 0 ; e -- ) { if ( arr [ e ] < arr [ e - 1 ] ) break ; }
max = arr [ s ] ; min = arr [ s ] ; for ( i = s + 1 ; i <= e ; i ++ ) { if ( arr [ i ] > max ) max = arr [ i ] ; if ( arr [ i ] < min ) min = arr [ i ] ; }
for ( i = 0 ; i < s ; i ++ ) { if ( arr [ i ] > min ) { s = i ; break ; } }
for ( i = n - 1 ; i >= e + 1 ; i -- ) { if ( arr [ i ] < max ) { e = i ; break ; } }
System . out . println ( " ▁ The ▁ unsorted ▁ subarray ▁ which " + " ▁ makes ▁ the ▁ given ▁ array ▁ sorted ▁ lies " + " ▁ between ▁ the ▁ indices ▁ " + s + " ▁ and ▁ " + e ) ; return ; } public static void main ( String args [ ] ) { int arr [ ] = { 10 , 12 , 20 , 30 , 25 , 40 , 32 , 31 , 35 , 50 , 60 } ; int arr_size = arr . length ; printUnsorted ( arr , arr_size ) ; } }
import java . io . * ; import java . util . * ; class CountTriangles {
static int findNumberOfTriangles ( int arr [ ] ) { int n = arr . length ;
Arrays . sort ( arr ) ;
int count = 0 ;
for ( int i = 0 ; i < n - 2 ; ++ i ) {
int k = i + 2 ;
for ( int j = i + 1 ; j < n ; ++ j ) {
while ( k < n && arr [ i ] + arr [ j ] > arr [ k ] ) ++ k ;
if ( k > j ) count += k - j - 1 ; } } return count ; }
public static void main ( String [ ] args ) { int arr [ ] = { 10 , 21 , 22 , 100 , 101 , 200 , 300 } ; System . out . println ( " Total ▁ number ▁ of ▁ triangles ▁ is ▁ " + findNumberOfTriangles ( arr ) ) ; } }
class Main {
static int findElement ( int arr [ ] , int n , int key ) { for ( int i = 0 ; i < n ; i ++ ) if ( arr [ i ] == key ) return i ; return - 1 ; }
public static void main ( String args [ ] ) { int arr [ ] = { 12 , 34 , 10 , 6 , 40 } ; int n = arr . length ;
int key = 40 ; int position = findElement ( arr , n , key ) ; if ( position == - 1 ) System . out . println ( " Element ▁ not ▁ found " ) ; else System . out . println ( " Element ▁ Found ▁ at ▁ Position : ▁ " + ( position + 1 ) ) ; } }
class Main {
static int insertSorted ( int arr [ ] , int n , int key , int capacity ) {
if ( n >= capacity ) return n ; arr [ n ] = key ; return ( n + 1 ) ; }
public static void main ( String [ ] args ) { int [ ] arr = new int [ 20 ] ; arr [ 0 ] = 12 ; arr [ 1 ] = 16 ; arr [ 2 ] = 20 ; arr [ 3 ] = 40 ; arr [ 4 ] = 50 ; arr [ 5 ] = 70 ; int capacity = 20 ; int n = 6 ; int i , key = 26 ; System . out . print ( " Before ▁ Insertion : ▁ " ) ; for ( i = 0 ; i < n ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ;
n = insertSorted ( arr , n , key , capacity ) ; System . out . print ( " After Insertion : "); for ( i = 0 ; i < n ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; } }
class Main {
static int findElement ( int arr [ ] , int n , int key ) { int i ; for ( i = 0 ; i < n ; i ++ ) if ( arr [ i ] == key ) return i ; return - 1 ; }
static int deleteElement ( int arr [ ] , int n , int key ) {
int pos = findElement ( arr , n , key ) ; if ( pos == - 1 ) { System . out . println ( " Element ▁ not ▁ found " ) ; return n ; }
int i ; for ( i = pos ; i < n - 1 ; i ++ ) arr [ i ] = arr [ i + 1 ] ; return n - 1 ; }
public static void main ( String args [ ] ) { int i ; int arr [ ] = { 10 , 50 , 30 , 40 , 20 } ; int n = arr . length ; int key = 30 ; System . out . println ( " Array ▁ before ▁ deletion " ) ; for ( i = 0 ; i < n ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; n = deleteElement ( arr , n , key ) ; System . out . println ( " Array after deletion "); for ( i = 0 ; i < n ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; } }
class Main {
static int binarySearch ( int arr [ ] , int low , int high , int key ) { if ( high < low ) return - 1 ;
int mid = ( low + high ) / 2 ; if ( key == arr [ mid ] ) return mid ; if ( key > arr [ mid ] ) return binarySearch ( arr , ( mid + 1 ) , high , key ) ; return binarySearch ( arr , low , ( mid - 1 ) , key ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 5 , 6 , 7 , 8 , 9 , 10 } ; int n , key ; n = arr . length - 1 ; key = 10 ; System . out . println ( " Index : ▁ " + binarySearch ( arr , 0 , n , key ) ) ; } }
class Main {
static int insertSorted ( int arr [ ] , int n , int key , int capacity ) {
if ( n >= capacity ) return n ; int i ; for ( i = n - 1 ; ( i >= 0 && arr [ i ] > key ) ; i -- ) arr [ i + 1 ] = arr [ i ] ; arr [ i + 1 ] = key ; return ( n + 1 ) ; }
public static void main ( String [ ] args ) { int arr [ ] = new int [ 20 ] ; arr [ 0 ] = 12 ; arr [ 1 ] = 16 ; arr [ 2 ] = 20 ; arr [ 3 ] = 40 ; arr [ 4 ] = 50 ; arr [ 5 ] = 70 ; int capacity = arr . length ; int n = 6 ; int key = 26 ; System . out . print ( " Before Insertion : "); for ( int i = 0 ; i < n ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ;
n = insertSorted ( arr , n , key , capacity ) ; System . out . print ( " After Insertion : "); for ( int i = 0 ; i < n ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; } }
class Main {
static int binarySearch ( int arr [ ] , int low , int high , int key ) { if ( high < low ) return - 1 ; int mid = ( low + high ) / 2 ; if ( key == arr [ mid ] ) return mid ; if ( key > arr [ mid ] ) return binarySearch ( arr , ( mid + 1 ) , high , key ) ; return binarySearch ( arr , low , ( mid - 1 ) , key ) ; }
static int deleteElement ( int arr [ ] , int n , int key ) {
int pos = binarySearch ( arr , 0 , n - 1 , key ) ; if ( pos == - 1 ) { System . out . println ( " Element ▁ not ▁ found " ) ; return n ; }
int i ; for ( i = pos ; i < n - 1 ; i ++ ) arr [ i ] = arr [ i + 1 ] ; return n - 1 ; }
public static void main ( String [ ] args ) { int i ; int arr [ ] = { 10 , 20 , 30 , 40 , 50 } ; int n = arr . length ; int key = 30 ; System . out . print ( "Array before deletion:NEW_LINE"); for ( i = 0 ; i < n ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; n = deleteElement ( arr , n , key ) ; System . out . print ( " Array after deletion : "); for ( i = 0 ; i < n ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; } }
class EquilibriumIndex {
int equilibrium ( int arr [ ] , int n ) { int i , j ; int leftsum , rightsum ;
for ( i = 0 ; i < n ; ++ i ) { leftsum = 0 ; rightsum = 0 ;
for ( j = 0 ; j < i ; j ++ ) leftsum += arr [ j ] ;
for ( j = i + 1 ; j < n ; j ++ ) rightsum += arr [ j ] ;
if ( leftsum == rightsum ) return i ; }
return - 1 ; }
public static void main ( String [ ] args ) { EquilibriumIndex equi = new EquilibriumIndex ( ) ; int arr [ ] = { - 7 , 1 , 5 , 2 , - 4 , 3 , 0 } ; int arr_size = arr . length ; System . out . println ( equi . equilibrium ( arr , arr_size ) ) ; } }
class EquilibriumIndex {
int equilibrium ( int arr [ ] , int n ) {
int sum = 0 ;
int leftsum = 0 ;
for ( int i = 0 ; i < n ; ++ i ) sum += arr [ i ] ; for ( int i = 0 ; i < n ; ++ i ) {
sum -= arr [ i ] ; if ( leftsum == sum ) return i ; leftsum += arr [ i ] ; }
return - 1 ; }
public static void main ( String [ ] args ) { EquilibriumIndex equi = new EquilibriumIndex ( ) ; int arr [ ] = { - 7 , 1 , 5 , 2 , - 4 , 3 , 0 } ; int arr_size = arr . length ; System . out . println ( " First ▁ equilibrium ▁ index ▁ is ▁ " + equi . equilibrium ( arr , arr_size ) ) ; } }
class Main {
static int ceilSearch ( int arr [ ] , int low , int high , int x ) { int i ;
if ( x <= arr [ low ] ) return low ;
for ( i = low ; i < high ; i ++ ) { if ( arr [ i ] == x ) return i ;
if ( arr [ i ] < x && arr [ i + 1 ] >= x ) return i + 1 ; }
return - 1 ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 8 , 10 , 10 , 12 , 19 } ; int n = arr . length ; int x = 3 ; int index = ceilSearch ( arr , 0 , n - 1 , x ) ; if ( index == - 1 ) System . out . println ( " Ceiling ▁ of ▁ " + x + " ▁ doesn ' t ▁ exist ▁ in ▁ array " ) ; else System . out . println ( " ceiling ▁ of ▁ " + x + " ▁ is ▁ " + arr [ index ] ) ; } }
class Main {
static int ceilSearch ( int arr [ ] , int low , int high , int x ) { int mid ;
if ( x <= arr [ low ] ) return low ;
if ( x > arr [ high ] ) return - 1 ;
mid = ( low + high ) / 2 ;
if ( arr [ mid ] == x ) return mid ;
else if ( arr [ mid ] < x ) { if ( mid + 1 <= high && x <= arr [ mid + 1 ] ) return mid + 1 ; else return ceilSearch ( arr , mid + 1 , high , x ) ; }
else { if ( mid - 1 >= low && x > arr [ mid - 1 ] ) return mid ; else return ceilSearch ( arr , low , mid - 1 , x ) ; } }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 8 , 10 , 10 , 12 , 19 } ; int n = arr . length ; int x = 8 ; int index = ceilSearch ( arr , 0 , n - 1 , x ) ; if ( index == - 1 ) System . out . println ( " Ceiling ▁ of ▁ " + x + " ▁ doesn ' t ▁ exist ▁ in ▁ array " ) ; else System . out . println ( " ceiling ▁ of ▁ " + x + " ▁ is ▁ " + arr [ index ] ) ; } }
import java . io . * ; class GFG { private static int isPairSum ( int A [ ] , int N , int X ) { for ( int i = 0 ; i < N ; i ++ ) { for ( int j = 0 ; j < N ; j ++ ) {
if ( i == j ) continue ;
if ( A [ i ] + A [ j ] == X ) return true ;
if ( A [ i ] + A [ j ] > X ) break ; } }
return 0 ; } }
public static void main ( String [ ] args ) { int arr [ ] = { 3 , 5 , 9 , 2 , 8 , 10 , 11 } ; int val = 17 ;
System . out . println ( isPairSum ( arr , arr . length , val ) ) ; }
import java . io . * ; class GFG {
public static int isPairSum ( int A [ ] , int N , int X ) {
int i = 0 ;
int j = N - 1 ; while ( i < j ) {
if ( A [ i ] + A [ j ] == X ) return 1 ;
else if ( A [ i ] + A [ j ] < X ) i ++ ;
else j -- ; } return 0 ; }
public static void main ( String [ ] args ) {
int arr [ ] = { 3 , 5 , 9 , 2 , 8 , 10 , 11 } ;
int val = 17 ;
int arrSize = arr . length ;
System . out . println ( isPairSum ( arr , arrSize , val ) ) ; } }
import java . io . * ; class GFG { static int NUM_LINE = 2 ; static int NUM_STATION = 4 ;
static int min ( int a , int b ) { return a < b ? a : b ; } static int carAssembly ( int a [ ] [ ] , int t [ ] [ ] , int e [ ] , int x [ ] ) { int T1 [ ] = new int [ NUM_STATION ] ; int T2 [ ] = new int [ NUM_STATION ] ; int i ;
T1 [ 0 ] = e [ 0 ] + a [ 0 ] [ 0 ] ;
T2 [ 0 ] = e [ 1 ] + a [ 1 ] [ 0 ] ;
for ( i = 1 ; i < NUM_STATION ; ++ i ) { T1 [ i ] = min ( T1 [ i - 1 ] + a [ 0 ] [ i ] , T2 [ i - 1 ] + t [ 1 ] [ i ] + a [ 0 ] [ i ] ) ; T2 [ i ] = min ( T2 [ i - 1 ] + a [ 1 ] [ i ] , T1 [ i - 1 ] + t [ 0 ] [ i ] + a [ 1 ] [ i ] ) ; }
return min ( T1 [ NUM_STATION - 1 ] + x [ 0 ] , T2 [ NUM_STATION - 1 ] + x [ 1 ] ) ; }
public static void main ( String [ ] args ) { int a [ ] [ ] = { { 4 , 5 , 3 , 2 } , { 2 , 10 , 1 , 4 } } ; int t [ ] [ ] = { { 0 , 7 , 4 , 5 } , { 0 , 9 , 2 , 8 } } ; int e [ ] = { 10 , 12 } , x [ ] = { 18 , 7 } ; System . out . println ( carAssembly ( a , t , e , x ) ) ; } }
import java . util . Arrays ; class GFG {
static int findMinInsertionsDP ( char str [ ] , int n ) {
int table [ ] [ ] = new int [ n ] [ n ] ; int l , h , gap ;
for ( gap = 1 ; gap < n ; ++ gap ) for ( l = 0 , h = gap ; h < n ; ++ l , ++ h ) table [ l ] [ h ] = ( str [ l ] == str [ h ] ) ? table [ l + 1 ] [ h - 1 ] : ( Integer . min ( table [ l ] [ h - 1 ] , table [ l + 1 ] [ h ] ) + 1 ) ;
return table [ 0 ] [ n - 1 ] ; }
public static void main ( String args [ ] ) { String str = " geeks " ; System . out . println ( findMinInsertionsDP ( str . toCharArray ( ) , str . length ( ) ) ) ; } }
class GFG {
static int max ( int x , int y ) { return ( x > y ) ? x : y ; }
static class Node { int data ; Node left , right ; } ;
static int LISS ( Node root ) { if ( root == null ) return 0 ;
int size_excl = LISS ( root . left ) + LISS ( root . right ) ;
int size_incl = 1 ; if ( root . left != null ) size_incl += LISS ( root . left . left ) + LISS ( root . left . right ) ; if ( root . right != null ) size_incl += LISS ( root . right . left ) + LISS ( root . right . right ) ;
return max ( size_incl , size_excl ) ; }
static Node newNode ( int data ) { Node temp = new Node ( ) ; temp . data = data ; temp . left = temp . right = null ; return temp ; }
public static void main ( String args [ ] ) {
Node root = newNode ( 20 ) ; root . left = newNode ( 8 ) ; root . left . left = newNode ( 4 ) ; root . left . right = newNode ( 12 ) ; root . left . right . left = newNode ( 10 ) ; root . left . right . right = newNode ( 14 ) ; root . right = newNode ( 22 ) ; root . right . right = newNode ( 25 ) ; System . out . println ( " Size ▁ of ▁ the ▁ Largest " + " ▁ Independent ▁ Set ▁ is ▁ " + LISS ( root ) ) ; } }
class Pair { int a ; int b ; public Pair ( int a , int b ) { this . a = a ; this . b = b ; }
static int maxChainLength ( Pair arr [ ] , int n ) { int i , j , max = 0 ; int mcl [ ] = new int [ n ] ;
for ( i = 0 ; i < n ; i ++ ) mcl [ i ] = 1 ;
for ( i = 1 ; i < n ; i ++ ) for ( j = 0 ; j < i ; j ++ ) if ( arr [ i ] . a > arr [ j ] . b && mcl [ i ] < mcl [ j ] + 1 ) mcl [ i ] = mcl [ j ] + 1 ;
for ( i = 0 ; i < n ; i ++ ) if ( max < mcl [ i ] ) max = mcl [ i ] ; return max ; }
public static void main ( String [ ] args ) { Pair arr [ ] = new Pair [ ] { new Pair ( 5 , 24 ) , new Pair ( 15 , 25 ) , new Pair ( 27 , 40 ) , new Pair ( 50 , 60 ) } ; System . out . println ( "Length of maximum size chain is " + maxChainLength ( arr , arr . length ) ) ; } }
public class GFG {
static int minPalPartion ( String str ) {
int n = str . length ( ) ;
int [ ] [ ] C = new int [ n ] [ n ] ; boolean [ ] [ ] P = new boolean [ n ] [ n ] ;
int i , j , k , L ;
for ( i = 0 ; i < n ; i ++ ) { P [ i ] [ i ] = true ; C [ i ] [ i ] = 0 ; }
for ( L = 2 ; L <= n ; L ++ ) {
for ( i = 0 ; i < n - L + 1 ; i ++ ) {
j = i + L - 1 ;
if ( L == 2 ) P [ i ] [ j ] = ( str . charAt ( i ) == str . charAt ( j ) ) ; else P [ i ] [ j ] = ( str . charAt ( i ) == str . charAt ( j ) ) && P [ i + 1 ] [ j - 1 ] ;
if ( P [ i ] [ j ] == true ) C [ i ] [ j ] = 0 ; else {
C [ i ] [ j ] = Integer . MAX_VALUE ; for ( k = i ; k <= j - 1 ; k ++ ) C [ i ] [ j ] = Integer . min ( C [ i ] [ j ] , C [ i ] [ k ] + C [ k + 1 ] [ j ] + 1 ) ; } } }
return C [ 0 ] [ n - 1 ] ; }
public static void main ( String args [ ] ) { String str = " ababbbabbababa " ; System . out . println ( " Min ▁ cuts ▁ needed ▁ for ▁ " + " Palindrome ▁ Partitioning ▁ is ▁ " + minPalPartion ( str ) ) ; } }
class AWQ { static int NO_OF_CHARS = 256 ;
static int max ( int a , int b ) { return ( a > b ) ? a : b ; }
static void badCharHeuristic ( char [ ] str , int size , int badchar [ ] ) {
for ( int i = 0 ; i < NO_OF_CHARS ; i ++ ) badchar [ i ] = - 1 ;
for ( int i = 0 ; i < size ; i ++ ) badchar [ ( int ) str [ i ] ] = i ; }
static void search ( char txt [ ] , char pat [ ] ) { int m = pat . length ; int n = txt . length ; int badchar [ ] = new int [ NO_OF_CHARS ] ;
badCharHeuristic ( pat , m , badchar ) ;
int s = 0 ; while ( s <= ( n - m ) ) { int j = m - 1 ;
while ( j >= 0 && pat [ j ] == txt [ s + j ] ) j -- ;
if ( j < 0 ) { System . out . println ( " Patterns ▁ occur ▁ at ▁ shift ▁ = ▁ " + s ) ;
s += ( s + m < n ) ? m - badchar [ txt [ s + m ] ] : 1 ; } else
s += max ( 1 , j - badchar [ txt [ s + j ] ] ) ; } }
public static void main ( String [ ] args ) { char txt [ ] = " ABAAABCD " . toCharArray ( ) ; char pat [ ] = " ABC " . toCharArray ( ) ; search ( txt , pat ) ; } }
class Node { int data ; Node left , right ; Node ( int item ) { data = item ; left = right ; } } class BinaryTree {
Node root ; int getLevelDiff ( Node node ) {
if ( node == null ) return 0 ;
return node . data - getLevelDiff ( node . left ) - getLevelDiff ( node . right ) ; }
public static void main ( String args [ ] ) { BinaryTree tree = new BinaryTree ( ) ; tree . root = new Node ( 5 ) ; tree . root . left = new Node ( 2 ) ; tree . root . right = new Node ( 6 ) ; tree . root . left . left = new Node ( 1 ) ; tree . root . left . right = new Node ( 4 ) ; tree . root . left . right . left = new Node ( 3 ) ; tree . root . right . right = new Node ( 8 ) ; tree . root . right . right . right = new Node ( 9 ) ; tree . root . right . right . left = new Node ( 7 ) ; System . out . println ( tree . getLevelDiff ( tree . root ) + " ▁ is ▁ the ▁ required ▁ difference " ) ; } }
class Node { int data ; Node left , right ; Node ( int item ) { data = item ; left = right = null ; } } class BinaryTree { Node root ;
boolean hasPathSum ( Node node , int sum ) { if ( node == null ) return sum == 0 ; return hasPathSum ( node . left , sum - node . data ) || hasPathSum ( node . right , sum - node . data ) ; }
public static void main ( String args [ ] ) { int sum = 21 ;
BinaryTree tree = new BinaryTree ( ) ; tree . root = new Node ( 10 ) ; tree . root . left = new Node ( 8 ) ; tree . root . right = new Node ( 2 ) ; tree . root . left . left = new Node ( 3 ) ; tree . root . left . right = new Node ( 5 ) ; tree . root . right . left = new Node ( 2 ) ; if ( tree . haspathSum ( tree . root , sum ) ) System . out . println ( " There ▁ is ▁ a ▁ root ▁ to ▁ leaf ▁ path ▁ with ▁ sum ▁ " + sum ) ; else System . out . println ( " There ▁ is ▁ no ▁ root ▁ to ▁ leaf ▁ path ▁ with ▁ sum ▁ " + sum ) ; } }
class Node { int data ; Node left , right ; Node ( int item ) { data = item ; left = right = null ; } } class BinaryTree { Node root ;
int treePathsSumUtil ( Node node , int val ) {
if ( node == null ) return 0 ;
val = ( val * 10 + node . data ) ;
if ( node . left == null && node . right == null ) return val ;
return treePathsSumUtil ( node . left , val ) + treePathsSumUtil ( node . right , val ) ; }
int treePathsSum ( Node node ) {
return treePathsSumUtil ( node , 0 ) ; }
public static void main ( String args [ ] ) { BinaryTree tree = new BinaryTree ( ) ; tree . root = new Node ( 6 ) ; tree . root . left = new Node ( 3 ) ; tree . root . right = new Node ( 5 ) ; tree . root . right . right = new Node ( 4 ) ; tree . root . left . left = new Node ( 2 ) ; tree . root . left . right = new Node ( 5 ) ; tree . root . left . right . right = new Node ( 4 ) ; tree . root . left . right . left = new Node ( 7 ) ; System . out . print ( " Sum ▁ of ▁ all ▁ paths ▁ is ▁ " + tree . treePathsSum ( tree . root ) ) ; } }
import java . util . HashMap ; import java . util . Map ;
class Node { int key ; Node left , right , parent ; Node ( int key ) { this . key = key ; left = right = parent = null ; } } class BinaryTree { Node root , n1 , n2 , lca ;
Node insert ( Node node , int key ) {
if ( node == null ) return new Node ( key ) ;
if ( key < node . key ) { node . left = insert ( node . left , key ) ; node . left . parent = node ; } else if ( key > node . key ) { node . right = insert ( node . right , key ) ; node . right . parent = node ; }
return node ; }
Node LCA ( Node n1 , Node n2 ) {
Map < Node , Boolean > ancestors = new HashMap < Node , Boolean > ( ) ;
while ( n1 != null ) { ancestors . put ( n1 , Boolean . TRUE ) ; n1 = n1 . parent ; }
while ( n2 != null ) { if ( ancestors . containsKey ( n2 ) != ancestors . isEmpty ( ) ) return n2 ; n2 = n2 . parent ; } return null ; }
public static void main ( String [ ] args ) { BinaryTree tree = new BinaryTree ( ) ; tree . root = tree . insert ( tree . root , 20 ) ; tree . root = tree . insert ( tree . root , 8 ) ; tree . root = tree . insert ( tree . root , 22 ) ; tree . root = tree . insert ( tree . root , 4 ) ; tree . root = tree . insert ( tree . root , 12 ) ; tree . root = tree . insert ( tree . root , 10 ) ; tree . root = tree . insert ( tree . root , 14 ) ; tree . n1 = tree . root . left . right . left ; tree . n2 = tree . root . left ; tree . lca = tree . LCA ( tree . n1 , tree . n2 ) ; System . out . println ( " LCA ▁ of ▁ " + tree . n1 . key + " ▁ and ▁ " + tree . n2 . key + " ▁ is ▁ " + tree . lca . key ) ; } }
import java . io . * ; class GFG {
public static int minChocolates ( int [ ] a , int n ) { int i = 0 , j = 0 ; int res = 0 , val = 1 ; while ( j < n - 1 ) { if ( a [ j ] > a [ j + 1 ] ) {
j += 1 ; continue ; } if ( i == j )
res += val ; else {
res += get_sum ( val , i , j ) ;
} if ( a [ j ] < a [ j + 1 ] )
val += 1 ; else
val = 1 ; j += 1 ; i = j ; }
if ( i == j ) res += val ; else res += get_sum ( val , i , j ) ; return res ; }
public static int get_sum ( int peak , int start , int end ) {
int count = end - start + 1 ;
peak = ( peak > count ) ? peak : count ;
int s = peak + ( ( ( count - 1 ) * count ) >> 1 ) ; return s ; } }
public static void main ( String [ ] args ) { int [ ] a = { 5 , 5 , 4 , 3 , 2 , 1 } ; int n = a . length ; System . out . print ( " Minimum ▁ number ▁ of ▁ chocolates ▁ = ▁ " + minChocolates ( a , n ) ) ; }
import java . io . * ; class GFG {
static double sum ( int n ) { double i , s = 0.0 ; for ( i = 1 ; i <= n ; i ++ ) s = s + 1 / i ; return s ; }
public static void main ( String args [ ] ) { int n = 5 ; System . out . printf ( " Sum ▁ is ▁ % f " , sum ( n ) ) ; } }
import java . lang . Math ; class GFG {
static long nthTermOfTheSeries ( int n ) {
long nthTerm ;
if ( n % 2 == 0 ) nthTerm = ( long ) Math . pow ( n - 1 , 2 ) + n ;
else nthTerm = ( long ) Math . pow ( n + 1 , 2 ) + n ;
return nthTerm ; }
public static void main ( String [ ] args ) { int n ; n = 8 ; System . out . println ( nthTermOfTheSeries ( n ) ) ; n = 12 ; System . out . println ( nthTermOfTheSeries ( n ) ) ; n = 102 ; System . out . println ( nthTermOfTheSeries ( n ) ) ; n = 999 ; System . out . println ( nthTermOfTheSeries ( n ) ) ; n = 9999 ; System . out . println ( nthTermOfTheSeries ( n ) ) ; } }
class Gfg1 { static int Log2n ( int n ) { return ( n > 1 ) ? 1 + Log2n ( n / 2 ) : 0 ; }
public static void main ( String args [ ] ) { int n = 32 ; System . out . println ( Log2n ( n ) ) ; } }
public class GFG { static float findAmount ( float X , float W , float Y ) { return ( X * ( Y - W ) ) / ( 100 - Y ) ; }
public static void main ( String args [ ] ) { float X = 100 , W = 50 , Y = 60 ; System . out . println ( " Water ▁ to ▁ be ▁ added ▁ = ▁ " + findAmount ( X , W , Y ) ) ; } }
import java . io . * ; public class GFG {
static float AvgofSquareN ( int n ) { return ( float ) ( ( n + 1 ) * ( 2 * n + 1 ) ) / 6 ; }
static public void main ( String [ ] args ) { int n = 2 ; System . out . println ( AvgofSquareN ( n ) ) ; } }
import java . util . * ; class GFG {
static void triangular_series ( int n ) { for ( int i = 1 ; i <= n ; i ++ ) System . out . printf ( " % d ▁ " ; , i * ( i + 1 ) / 2 ) ; }
public static void main ( String [ ] args ) { int n = 5 ; triangular_series ( n ) ; } }
import java . io . * ; class GFG {
static int divisorSum ( int n ) { int sum = 0 ; for ( int i = 1 ; i <= n ; ++ i ) sum += ( n / i ) * i ; return sum ; }
public static void main ( String args [ ] ) { int n = 4 ; System . out . println ( divisorSum ( n ) ) ; n = 5 ; System . out . println ( divisorSum ( n ) ) ; } }
import static java . lang . Math . pow ; class GFG {
static double sum ( int x , int n ) { double i , total = 1.0 ; for ( i = 1 ; i <= n ; i ++ ) total = total + ( Math . pow ( x , i ) / i ) ; return total ; }
public static void main ( String [ ] args ) { int x = 2 ; int n = 5 ; System . out . printf ( " % .2f " , sum ( x , n ) ) ; } }
public class Power_3 {
static boolean check ( int n ) { if ( n <= 0 ) return false ;
return 1162261467 % n == 0 ; }
public static void main ( String args [ ] ) { int n = 9 ; if ( check ( n ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
import java . io . * ; class GFG { static int per ( int n ) { int a = 3 , b = 0 , c = 2 , i ; int m = 0 ; if ( n == 0 ) return a ; if ( n == 1 ) return b ; if ( n == 2 ) return c ; while ( n > 2 ) { m = a + b ; a = b ; b = c ; c = m ; n -- ; } return m ; }
public static void main ( String [ ] args ) { int n = 9 ; System . out . println ( per ( n ) ) ; } }
import java . io . * ; import java . math . * ; class GFG {
static void countDivisors ( int n ) {
int count = 0 ;
for ( int i = 1 ; i <= Math . sqrt ( n ) + 1 ; i ++ ) { if ( n % i == 0 )
count += ( n / i == i ) ? 1 : 2 ; } if ( count % 2 == 0 ) System . out . println ( " Even " ) ; else System . out . println ( " Odd " ) ; }
public static void main ( String args [ ] ) { System . out . print ( " The ▁ count ▁ of ▁ divisor : ▁ " ) ; countDivisors ( 10 ) ; } }
static long multiply ( long a , long b , long mod ) { return ( ( a % mod ) * ( b % mod ) ) % mod ; }
class GFG {
static int countSquares ( int m , int n ) {
if ( n < m ) { int temp = m ; m = n ; n = temp ; }
return m * ( m + 1 ) * ( 2 * m + 1 ) / 6 + ( n - m ) * m * ( m + 1 ) / 2 ; }
public static void main ( String [ ] args ) { int m = 4 , n = 3 ; System . out . println ( " Count ▁ of ▁ squares ▁ is ▁ " + countSquares ( m , n ) ) ; } }
import java . io . * ; class GFG {
static double sum ( int n ) { double i , s = 0.0 ; for ( i = 1 ; i <= n ; i ++ ) s = s + 1 / i ; return s ; }
public static void main ( String args [ ] ) { int n = 5 ; System . out . printf ( " Sum ▁ is ▁ % f " , sum ( n ) ) ; } }
class Test {
static int gcd ( int a , int b ) { if ( b == 0 ) return a ; return gcd ( b , a % b ) ; }
public static void main ( String [ ] args ) { int a = 98 , b = 56 ; System . out . println ( " GCD ▁ of ▁ " + a + " ▁ and ▁ " + b + " ▁ is ▁ " + gcd ( a , b ) ) ; } }
class GfG {
static void printArray ( int arr [ ] , int size ) { for ( int i = 0 ; i < size ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; System . out . println ( ) ; return ; }
static void printSequencesRecur ( int arr [ ] , int n , int k , int index ) { int i ; if ( k == 0 ) { printArray ( arr , index ) ; } if ( k > 0 ) { for ( i = 1 ; i <= n ; ++ i ) { arr [ index ] = i ; printSequencesRecur ( arr , n , k - 1 , index + 1 ) ; } } }
static void printSequences ( int n , int k ) { int arr [ ] = new int [ k ] ; printSequencesRecur ( arr , n , k , 0 ) ; return ; }
public static void main ( String [ ] args ) { int n = 3 ; int k = 2 ; printSequences ( n , k ) ; } }
class GFG {
static boolean isMultipleof5 ( int n ) { while ( n > 0 ) n = n - 5 ; if ( n == 0 ) return true ; return false ; }
public static void main ( String [ ] args ) { int n = 19 ; if ( isMultipleof5 ( n ) == true ) System . out . printf ( "%d is multiple of 5NEW_LINE", n); else System . out . printf ( "%d is not a multiple of 5NEW_LINE", n); } }
class GFG { static int countBits ( int n ) { int count = 0 ; while ( n != 0 ) { count ++ ; n >>= 1 ; } return count ; }
public static void main ( String [ ] arg ) { int i = 65 ; System . out . print ( countBits ( i ) ) ; } }
import java . io . * ; class GFG { static int INT_MAX = 2147483647 ;
static int isKthBitSet ( int x , int k ) { return ( ( x & ( 1 << ( k - 1 ) ) ) > 0 ) ? 1 : 0 ; }
static int leftmostSetBit ( int x ) { int count = 0 ; while ( x > 0 ) { count ++ ; x = x >> 1 ; } return count ; }
static int isBinPalindrome ( int x ) { int l = leftmostSetBit ( x ) ; int r = 1 ;
while ( l > r ) {
if ( isKthBitSet ( x , l ) != isKthBitSet ( x , r ) ) return 0 ; l -- ; r ++ ; } return 1 ; } static int findNthPalindrome ( int n ) { int pal_count = 0 ;
int i = 0 ; for ( i = 1 ; i <= INT_MAX ; i ++ ) { if ( isBinPalindrome ( i ) > 0 ) { pal_count ++ ; }
if ( pal_count == n ) break ; } return i ; }
public static void main ( String [ ] args ) { int n = 9 ;
System . out . println ( findNthPalindrome ( n ) ) ; } }
import java . io . * ; class GFG {
static double temp_convert ( int F1 , int B1 , int F2 , int B2 , int T ) { float t2 ;
t2 = F2 + ( float ) ( B2 - F2 ) / ( B1 - F1 ) * ( T - F1 ) ; return t2 ; }
public static void main ( String [ ] args ) { int F1 = 0 , B1 = 100 ; int F2 = 32 , B2 = 212 ; int T = 37 ; float t2 ; System . out . printf ( " % .2f " , temp_convert ( F1 , B1 , F2 , B2 , T ) ) ; } }
class Node { int data ; Node left , right ; Node ( int item ) { data = item ; left = right = null ; } } class BinaryTree { Node root ;
int maxDepth ( Node node ) { if ( node == null ) return 0 ; else {
int lDepth = maxDepth ( node . left ) ; int rDepth = maxDepth ( node . right ) ;
if ( lDepth > rDepth ) return ( lDepth + 1 ) ; else return ( rDepth + 1 ) ; } }
public static void main ( String [ ] args ) { BinaryTree tree = new BinaryTree ( ) ; tree . root = new Node ( 1 ) ; tree . root . left = new Node ( 2 ) ; tree . root . right = new Node ( 3 ) ; tree . root . left . left = new Node ( 4 ) ; tree . root . left . right = new Node ( 5 ) ; System . out . println ( " Height ▁ of ▁ tree ▁ is ▁ : ▁ " + tree . maxDepth ( tree . root ) ) ; } }
class Node { int data ; Node left , right ; Node ( int d ) { data = d ; left = right = null ; } } class BinaryTree { Node root ;
boolean isBalanced ( Node node ) {
int lh ;
int rh ;
if ( node == null ) return true ;
lh = height ( node . left ) ; rh = height ( node . right ) ; if ( Math . abs ( lh - rh ) <= 1 && isBalanced ( node . left ) && isBalanced ( node . right ) ) return true ;
return false ; }
int height ( Node node ) {
if ( node == null ) return 0 ;
return 1 + Math . max ( height ( node . left ) , height ( node . right ) ) ; }
public static void main ( String args [ ] ) { BinaryTree tree = new BinaryTree ( ) ; tree . root = new Node ( 1 ) ; tree . root . left = new Node ( 2 ) ; tree . root . right = new Node ( 3 ) ; tree . root . left . left = new Node ( 4 ) ; tree . root . left . right = new Node ( 5 ) ; tree . root . left . left . left = new Node ( 8 ) ; if ( tree . isBalanced ( tree . root ) ) System . out . println ( " Tree ▁ is ▁ balanced " ) ; else System . out . println ( " Tree ▁ is ▁ not ▁ balanced " ) ; } }
class Node { int data ; Node left , right ; public Node ( int item ) { data = item ; left = right = null ; } }
static int height ( Node node ) {
if ( node == null ) return 0 ;
return ( 1 + Math . max ( height ( node . left ) , height ( node . right ) ) ) ; }
int diameter ( Node root ) {
if ( root == null ) return 0 ;
int lheight = height ( root . left ) ; int rheight = height ( root . right ) ;
int ldiameter = diameter ( root . left ) ; int rdiameter = diameter ( root . right ) ;
return Math . max ( lheight + rheight + 1 , Math . max ( ldiameter , rdiameter ) ) ; }
public static void main ( String args [ ] ) {
BinaryTree tree = new BinaryTree ( ) ; tree . root = new Node ( 1 ) ; tree . root . left = new Node ( 2 ) ; tree . root . right = new Node ( 3 ) ; tree . root . left . left = new Node ( 4 ) ; tree . root . left . right = new Node ( 5 ) ;
System . out . println ( " The ▁ diameter ▁ of ▁ given ▁ binary ▁ tree ▁ is ▁ : ▁ " + tree . diameter ( ) ) ; } }
import java . io . * ; class GFG {
static void findpath ( int N , int a [ ] ) {
if ( a [ 0 ] == 1 ) {
System . out . print ( ( N + 1 ) + " ▁ " ) ; for ( int i = 1 ; i <= N ; i ++ ) System . out . print ( ( i ) + " ▁ " ) ; return ; }
for ( int i = 0 ; i < N - 1 ; i ++ ) { if ( a [ i ] == 0 && a [ i + 1 ] == 1 ) {
for ( int j = 1 ; j <= i ; j ++ ) System . out . print ( ( j ) + " ▁ " ) ; System . out . print ( ( N + 1 ) + " ▁ " ) ; for ( int j = i + 1 ; j <= N ; j ++ ) System . out . print ( ( j ) + " ▁ " ) ; return ; } }
for ( int i = 1 ; i <= N ; i ++ ) System . out . print ( ( i ) + " ▁ " ) ; System . out . print ( ( N + 1 ) + " ▁ " ) ; }
public static void main ( String [ ] args ) { int N = 3 , arr [ ] = { 0 , 1 , 0 } ;
int N = 3 , arr [ ] = { 0 , 1 , 0 } ;
findpath ( N , arr ) ; } }
class Node { int data ; Node left , right ; Node ( int item ) { data = item ; left = right = null ; } } class BinaryTree { Node root ;
int depthOfOddLeafUtil ( Node node , int level ) {
if ( node == null ) return 0 ;
if ( node . left == null && node . right == null && ( level & 1 ) != 0 ) return level ;
return Math . max ( depthOfOddLeafUtil ( node . left , level + 1 ) , depthOfOddLeafUtil ( node . right , level + 1 ) ) ; }
int depthOfOddLeaf ( Node node ) { int level = 1 , depth = 0 ; return depthOfOddLeafUtil ( node , level ) ; }
public static void main ( String args [ ] ) { int k = 45 ; BinaryTree tree = new BinaryTree ( ) ; tree . root = new Node ( 1 ) ; tree . root . left = new Node ( 2 ) ; tree . root . right = new Node ( 3 ) ; tree . root . left . left = new Node ( 4 ) ; tree . root . right . left = new Node ( 5 ) ; tree . root . right . right = new Node ( 6 ) ; tree . root . right . left . right = new Node ( 7 ) ; tree . root . right . right . right = new Node ( 8 ) ; tree . root . right . left . right . left = new Node ( 9 ) ; tree . root . right . right . right . right = new Node ( 10 ) ; tree . root . right . right . right . right . left = new Node ( 11 ) ; System . out . println ( tree . depthOfOddLeaf ( tree . root ) + " ▁ is ▁ the ▁ required ▁ depth " ) ; } }
import java . util . * ; class GFG {
public static void printArr ( int [ ] arr , int n ) {
Arrays . sort ( arr ) ;
if ( arr [ 0 ] == arr [ n - 1 ] ) { System . out . println ( " No " ) ; }
else { System . out . println ( " Yes " ) ; for ( int i = 0 ; i < n ; i ++ ) { System . out . print ( arr [ i ] + " ▁ " ) ; } } }
public static void main ( String [ ] args ) {
int arr [ ] = { 1 , 2 , 2 , 1 , 3 , 1 } ; int N = arr . length ;
printArr ( arr , N ) ; } }
class Node { int data ; Node left , right ; Node ( int item ) { data = item ; left = right = null ; } } class BinaryTree { Node root ;
int getMaxWidth ( Node node ) { int maxWidth = 0 ; int width ; int h = height ( node ) ; int i ;
for ( i = 1 ; i <= h ; i ++ ) { width = getWidth ( node , i ) ; if ( width > maxWidth ) maxWidth = width ; } return maxWidth ; }
int getWidth ( Node node , int level ) { if ( node == null ) return 0 ; if ( level == 1 ) return 1 ; else if ( level > 1 ) return getWidth ( node . left , level - 1 ) + getWidth ( node . right , level - 1 ) ; return 0 ; }
int height ( Node node ) { if ( node == null ) return 0 ; else {
int lHeight = height ( node . left ) ; int rHeight = height ( node . right ) ;
return ( lHeight > rHeight ) ? ( lHeight + 1 ) : ( rHeight + 1 ) ; } }
public static void main ( String args [ ] ) { BinaryTree tree = new BinaryTree ( ) ;
tree . root = new Node ( 1 ) ; tree . root . left = new Node ( 2 ) ; tree . root . right = new Node ( 3 ) ; tree . root . left . left = new Node ( 4 ) ; tree . root . left . right = new Node ( 5 ) ; tree . root . right . right = new Node ( 8 ) ; tree . root . right . right . left = new Node ( 6 ) ; tree . root . right . right . right = new Node ( 7 ) ;
System . out . println ( " Maximum ▁ width ▁ is ▁ " + tree . getMaxWidth ( tree . root ) ) ; } }
class Node { int data ; Node left , right ; Node ( int item ) { data = item ; left = right = null ; } } class BinaryTree { Node root ;
int getMaxWidth ( Node node ) { int width ; int h = height ( node ) ;
int count [ ] = new int [ 10 ] ; int level = 0 ;
getMaxWidthRecur ( node , count , level ) ;
return getMax ( count , h ) ; }
void getMaxWidthRecur ( Node node , int count [ ] , int level ) { if ( node != null ) { count [ level ] ++ ; getMaxWidthRecur ( node . left , count , level + 1 ) ; getMaxWidthRecur ( node . right , count , level + 1 ) ; } }
int height ( Node node ) { if ( node == null ) return 0 ; else {
int lHeight = height ( node . left ) ; int rHeight = height ( node . right ) ;
return ( lHeight > rHeight ) ? ( lHeight + 1 ) : ( rHeight + 1 ) ; } }
int getMax ( int arr [ ] , int n ) { int max = arr [ 0 ] ; int i ; for ( i = 0 ; i < n ; i ++ ) { if ( arr [ i ] > max ) max = arr [ i ] ; } return max ; }
public static void main ( String args [ ] ) { BinaryTree tree = new BinaryTree ( ) ;
tree . root = new Node ( 1 ) ; tree . root . left = new Node ( 2 ) ; tree . root . right = new Node ( 3 ) ; tree . root . left . left = new Node ( 4 ) ; tree . root . left . right = new Node ( 5 ) ; tree . root . right . right = new Node ( 8 ) ; tree . root . right . right . left = new Node ( 6 ) ; tree . root . right . right . right = new Node ( 7 ) ; System . out . println ( " Maximum ▁ width ▁ is ▁ " + tree . getMaxWidth ( tree . root ) ) ; } }
class Node { int data ; Node left , right ; public Node ( int item ) { data = item ; left = right = null ; } } public class BinaryTree {
int getLeafCount ( ) { return getLeafCount ( root ) ; } int getLeafCount ( Node node ) { if ( node == null ) return 0 ; if ( node . left == null && node . right == null ) return 1 ; else return getLeafCount ( node . left ) + getLeafCount ( node . right ) ; }
public static void main ( String args [ ] ) {
BinaryTree tree = new BinaryTree ( ) ; tree . root = new Node ( 1 ) ; tree . root . left = new Node ( 2 ) ; tree . root . right = new Node ( 3 ) ; tree . root . left . left = new Node ( 4 ) ; tree . root . left . right = new Node ( 5 ) ;
System . out . println ( " The ▁ leaf ▁ count ▁ of ▁ binary ▁ tree ▁ is ▁ : ▁ " + tree . getLeafCount ( ) ) ; } }
import java . util . Vector ; class GFG {
static int deno [ ] = { 1 , 2 , 5 , 10 , 20 , 50 , 100 , 500 , 1000 } ; static int n = deno . length ; static void findMin ( int V ) {
Vector < Integer > ans = new Vector < > ( ) ;
for ( int i = n - 1 ; i >= 0 ; i -- ) {
while ( V >= deno [ i ] ) { V -= deno [ i ] ; ans . add ( deno [ i ] ) ; } }
for ( int i = 0 ; i < ans . size ( ) ; i ++ ) { System . out . print ( " ▁ " + ans . elementAt ( i ) ) ; } }
public static void main ( String [ ] args ) { int n = 93 ; System . out . print ( " Following ▁ is ▁ minimal ▁ number ▁ " + " of ▁ change ▁ for ▁ " + n + " : ▁ " ) ; findMin ( n ) ; } }
class Node { int data ; Node left , right , nextRight ; Node ( int item ) { data = item ; left = right = nextRight = null ; } } class BinaryTree { Node root ;
Node getNextRight ( Node p ) { Node temp = p . nextRight ;
while ( temp != null ) { if ( temp . left != null ) return temp . left ; if ( temp . right != null ) return temp . right ; temp = temp . nextRight ; }
return null ; }
void connect ( Node p ) { Node temp = null ; if ( p == null ) return ;
p . nextRight = null ;
while ( p != null ) { Node q = p ;
while ( q != null ) {
if ( q . left != null ) {
if ( q . right != null ) q . left . nextRight = q . right ; else q . left . nextRight = getNextRight ( q ) ; } if ( q . right != null ) q . right . nextRight = getNextRight ( q ) ;
q = q . nextRight ; }
if ( p . left != null ) p = p . left ; else if ( p . right != null ) p = p . right ; else p = getNextRight ( p ) ; } }
public static void main ( String args [ ] ) {
BinaryTree tree = new BinaryTree ( ) ; tree . root = new Node ( 10 ) ; tree . root . left = new Node ( 8 ) ; tree . root . right = new Node ( 2 ) ; tree . root . left . left = new Node ( 3 ) ; tree . root . right . right = new Node ( 90 ) ;
tree . connect ( tree . root ) ;
int a = tree . root . nextRight != null ? tree . root . nextRight . data : - 1 ; int b = tree . root . left . nextRight != null ? tree . root . left . nextRight . data : - 1 ; int c = tree . root . right . nextRight != null ? tree . root . right . nextRight . data : - 1 ; int d = tree . root . left . left . nextRight != null ? tree . root . left . left . nextRight . data : - 1 ; int e = tree . root . right . right . nextRight != null ? tree . root . right . right . nextRight . data : - 1 ; System . out . println ( " Following ▁ are ▁ populated ▁ nextRight ▁ pointers ▁ in ▁ " + " ▁ the ▁ tree ( -1 ▁ is ▁ printed ▁ if ▁ there ▁ is ▁ no ▁ nextRight ) " ) ; System . out . println ( " nextRight ▁ of ▁ " + tree . root . data + " ▁ is ▁ " + a ) ; System . out . println ( " nextRight ▁ of ▁ " + tree . root . left . data + " ▁ is ▁ " + b ) ; System . out . println ( " nextRight ▁ of ▁ " + tree . root . right . data + " ▁ is ▁ " + c ) ; System . out . println ( " nextRight ▁ of ▁ " + tree . root . left . left . data + " ▁ is ▁ " + d ) ; System . out . println ( " nextRight ▁ of ▁ " + tree . root . right . right . data + " ▁ is ▁ " + e ) ; } }
class Node { int data ; Node left , right , nextRight ; Node ( int item ) { data = item ; left = right = nextRight = null ; } } class BinaryTree { Node root ;
void connect ( Node p ) {
p . nextRight = null ;
connectRecur ( p ) ; }
void connectRecur ( Node p ) {
if ( p == null ) return ;
if ( p . left != null ) p . left . nextRight = p . right ;
if ( p . right != null ) p . right . nextRight = ( p . nextRight != null ) ? p . nextRight . left : null ;
connectRecur ( p . left ) ; connectRecur ( p . right ) ; }
public static void main ( String args [ ] ) { BinaryTree tree = new BinaryTree ( ) ;
tree . root = new Node ( 10 ) ; tree . root . left = new Node ( 8 ) ; tree . root . right = new Node ( 2 ) ; tree . root . left . left = new Node ( 3 ) ;
tree . connect ( tree . root ) ;
System . out . println ( " Following ▁ are ▁ populated ▁ nextRight ▁ pointers ▁ in ▁ " + " the ▁ tree " + " ( -1 ▁ is ▁ printed ▁ if ▁ there ▁ is ▁ no ▁ nextRight ) " ) ; int a = tree . root . nextRight != null ? tree . root . nextRight . data : - 1 ; System . out . println ( " nextRight ▁ of ▁ " + tree . root . data + " ▁ is ▁ " + a ) ; int b = tree . root . left . nextRight != null ? tree . root . left . nextRight . data : - 1 ; System . out . println ( " nextRight ▁ of ▁ " + tree . root . left . data + " ▁ is ▁ " + b ) ; int c = tree . root . right . nextRight != null ? tree . root . right . nextRight . data : - 1 ; System . out . println ( " nextRight ▁ of ▁ " + tree . root . right . data + " ▁ is ▁ " + c ) ; int d = tree . root . left . left . nextRight != null ? tree . root . left . left . nextRight . data : - 1 ; System . out . println ( " nextRight ▁ of ▁ " + tree . root . left . left . data + " ▁ is ▁ " + d ) ; } }
class GFG {
static int findMinInsertions ( char str [ ] , int l , int h ) {
if ( l > h ) return Integer . MAX_VALUE ; if ( l == h ) return 0 ; if ( l == h - 1 ) return ( str [ l ] == str [ h ] ) ? 0 : 1 ;
return ( str [ l ] == str [ h ] ) ? findMinInsertions ( str , l + 1 , h - 1 ) : ( Integer . min ( findMinInsertions ( str , l , h - 1 ) , findMinInsertions ( str , l + 1 , h ) ) + 1 ) ; }
public static void main ( String args [ ] ) { String str = " geeks " ; System . out . println ( findMinInsertions ( str . toCharArray ( ) , 0 , str . length ( ) - 1 ) ) ; } }
class GFG {
static int max ( int x , int y ) { return ( x > y ) ? x : y ; }
static int lps ( char seq [ ] , int i , int j ) {
if ( i == j ) { return 1 ; }
if ( seq [ i ] == seq [ j ] && i + 1 == j ) { return 2 ; }
if ( seq [ i ] == seq [ j ] ) { return lps ( seq , i + 1 , j - 1 ) + 2 ; }
return max ( lps ( seq , i , j - 1 ) , lps ( seq , i + 1 , j ) ) ; }
public static void main ( String [ ] args ) { String seq = " GEEKSFORGEEKS " ; int n = seq . length ( ) ; System . out . printf ( " The ▁ length ▁ of ▁ the ▁ LPS ▁ is ▁ % d " , lps ( seq . toCharArray ( ) , 0 , n - 1 ) ) ; } }
class Node { int data ; Node left , right ; public Node ( int d ) { data = d ; left = right = null ; } } class BinaryTree { Node root ;
int getLevelUtil ( Node node , int data , int level ) { if ( node == null ) return 0 ; if ( node . data == data ) return level ; int downlevel = getLevelUtil ( node . left , data , level + 1 ) ; if ( downlevel != 0 ) return downlevel ; downlevel = getLevelUtil ( node . right , data , level + 1 ) ; return downlevel ; }
int getLevel ( Node node , int data ) { return getLevelUtil ( node , data , 1 ) ; }
public static void main ( String [ ] args ) { BinaryTree tree = new BinaryTree ( ) ;
tree . root = new Node ( 3 ) ; tree . root . left = new Node ( 2 ) ; tree . root . right = new Node ( 5 ) ; tree . root . left . left = new Node ( 1 ) ; tree . root . left . right = new Node ( 4 ) ; for ( int x = 1 ; x <= 5 ; x ++ ) { int level = tree . getLevel ( tree . root , x ) ; if ( level != 0 ) System . out . println ( " Level ▁ of ▁ " + x + " ▁ is ▁ " + tree . getLevel ( tree . root , x ) ) ; else System . out . println ( x + " ▁ is ▁ not ▁ present ▁ in ▁ tree " ) ; } } }
class GFG { static int NO_OF_CHARS = 256 ; static int getNextState ( char [ ] pat , int M , int state , int x ) {
if ( state < M && x == pat [ state ] ) return state + 1 ;
int ns , i ;
for ( ns = state ; ns > 0 ; ns -- ) { if ( pat [ ns - 1 ] == x ) { for ( i = 0 ; i < ns - 1 ; i ++ ) if ( pat [ i ] != pat [ state - ns + 1 + i ] ) break ; if ( i == ns - 1 ) return ns ; } } return 0 ; }
static void computeTF ( char [ ] pat , int M , int TF [ ] [ ] ) { int state , x ; for ( state = 0 ; state <= M ; ++ state ) for ( x = 0 ; x < NO_OF_CHARS ; ++ x ) TF [ state ] [ x ] = getNextState ( pat , M , state , x ) ; }
static void search ( char [ ] pat , char [ ] txt ) { int M = pat . length ; int N = txt . length ; int [ ] [ ] TF = new int [ M + 1 ] [ NO_OF_CHARS ] ; computeTF ( pat , M , TF ) ;
int i , state = 0 ; for ( i = 0 ; i < N ; i ++ ) { state = TF [ state ] [ txt [ i ] ] ; if ( state == M ) System . out . println ( " Pattern ▁ found ▁ " + " at ▁ index ▁ " + ( i - M + 1 ) ) ; } }
public static void main ( String [ ] args ) { char [ ] pat = " AABAACAADAABAAABAA " . toCharArray ( ) ; char [ ] txt = " AABA " . toCharArray ( ) ; search ( txt , pat ) ; } }
class GfG {
static class Node { int key ; Node left , right ; }
static Node newNode ( int key ) { Node n = new Node ( ) ; n . key = key ; n . left = null ; n . right = null ; return n ; }
static int findMirrorRec ( int target , Node left , Node right ) {
if ( left == null right == null ) return 0 ;
if ( left . key == target ) return right . key ; if ( right . key == target ) return left . key ;
int mirror_val = findMirrorRec ( target , left . left , right . right ) ; if ( mirror_val != 0 ) return mirror_val ;
return findMirrorRec ( target , left . right , right . left ) ; }
static int findMirror ( Node root , int target ) { if ( root == null ) return 0 ; if ( root . key == target ) return target ; return findMirrorRec ( target , root . left , root . right ) ; }
public static void main ( String [ ] args ) { Node root = newNode ( 1 ) ; root . left = newNode ( 2 ) ; root . left . left = newNode ( 4 ) ; root . left . left . right = newNode ( 7 ) ; root . right = newNode ( 3 ) ; root . right . left = newNode ( 5 ) ; root . right . right = newNode ( 6 ) ; root . right . left . left = newNode ( 8 ) ; root . right . left . right = newNode ( 9 ) ;
int target = root . left . left . key ; int mirror = findMirror ( root , target ) ; if ( mirror != 0 ) System . out . println ( " Mirror ▁ of ▁ Node ▁ " + target + " ▁ is ▁ Node ▁ " + mirror ) ; else System . out . println ( " Mirror ▁ of ▁ Node ▁ " + target + " ▁ is ▁ null ▁ " ) ; } }
import java . util . * ; class GFG {
static class node { int data ; node left ; node right ;
node ( int data ) { this . data = data ; this . left = null ; this . right = null ; } } ;
static boolean iterativeSearch ( node root , int x ) {
if ( root == null ) return false ;
Queue < node > q = new LinkedList ( ) ;
q . add ( root ) ;
while ( q . size ( ) > 0 ) {
node node = q . peek ( ) ; if ( node . data == x ) return true ;
q . remove ( ) ; if ( node . left != null ) q . add ( node . left ) ; if ( node . right != null ) q . add ( node . right ) ; } return false ; }
public static void main ( String ags [ ] ) { node NewRoot = null ; node root = new node ( 2 ) ; root . left = new node ( 7 ) ; root . right = new node ( 5 ) ; root . left . right = new node ( 6 ) ; root . left . right . left = new node ( 1 ) ; root . left . right . right = new node ( 11 ) ; root . right . right = new node ( 9 ) ; root . right . right . left = new node ( 4 ) ; System . out . print ( ( iterativeSearch ( root , 6 ) ? "Found " : ▁ " Not Found "));  System . out . print ( ( iterativeSearch ( root , 12 ) ? "Found " : ▁ " Not Found "));  } }
class Node { int data ; Node left , right , next ; Node ( int item ) { data = item ; left = right = next = null ; } }
void populateNext ( Node node ) {
populateNextRecur ( node , next ) ; }
void populateNextRecur ( Node p , Node next_ref ) { if ( p != null ) {
populateNextRecur ( p . right , next_ref ) ;
p . next = next_ref ;
next_ref = p ;
populateNextRecur ( p . left , next_ref ) ; } }
class Node { int data ; Node left , right ; Node ( int item ) { data = item ; left = right = null ; } }
class BinaryTree { Node root ; void printInorder ( Node node ) { if ( node != null ) { printInorder ( node . left ) ; System . out . print ( node . data + " ▁ " ) ; printInorder ( node . right ) ; } }
Node RemoveHalfNodes ( Node node ) { if ( node == null ) return null ; node . left = RemoveHalfNodes ( node . left ) ; node . right = RemoveHalfNodes ( node . right ) ; if ( node . left == null && node . right == null ) return node ;
if ( node . left == null ) { Node new_root = node . right ; return new_root ; }
if ( node . right == null ) { Node new_root = node . left ; return new_root ; } return node ; }
public static void main ( String args [ ] ) { BinaryTree tree = new BinaryTree ( ) ; Node NewRoot = null ; tree . root = new Node ( 2 ) ; tree . root . left = new Node ( 7 ) ; tree . root . right = new Node ( 5 ) ; tree . root . left . right = new Node ( 6 ) ; tree . root . left . right . left = new Node ( 1 ) ; tree . root . left . right . right = new Node ( 11 ) ; tree . root . right . right = new Node ( 9 ) ; tree . root . right . right . left = new Node ( 4 ) ; System . out . println ( " the ▁ inorder ▁ traversal ▁ of ▁ tree ▁ is ▁ " ) ; tree . printInorder ( tree . root ) ; NewRoot = tree . RemoveHalfNodes ( tree . root ) ; System . out . print ( " Inorder traversal of the modified tree "); tree . printInorder ( NewRoot ) ; } }
import java . io . * ; class GFG { public static void printSubstrings ( String str ) {
for ( int i = 0 ; i < n ; i ++ ) {
for ( int j = i ; j < n ; j ++ ) {
for ( int k = i ; k <= j ; k ++ ) { System . out . print ( str . charAt ( k ) ) ; }
System . out . println ( ) ; } } }
public static void main ( String [ ] args ) { String str = " abcd " ;
printSubstrings ( str ) ; } }
public class Suduko {
static int N = 9 ;
static void print ( int [ ] [ ] grid ) { for ( int i = 0 ; i < N ; i ++ ) { for ( int j = 0 ; j < N ; j ++ ) System . out . print ( grid [ i ] [ j ] + " ▁ " ) ; System . out . println ( ) ; } }
static boolean isSafe ( int [ ] [ ] grid , int row , int col , int num ) {
for ( int x = 0 ; x <= 8 ; x ++ ) if ( grid [ row ] [ x ] == num ) return false ;
for ( int x = 0 ; x <= 8 ; x ++ ) if ( grid [ x ] [ col ] == num ) return false ;
int startRow = row - row % 3 , startCol = col - col % 3 ; for ( int i = 0 ; i < 3 ; i ++ ) for ( int j = 0 ; j < 3 ; j ++ ) if ( grid [ i + startRow ] [ j + startCol ] == num ) return false ; return true ; }
static boolean solveSuduko ( int grid [ ] [ ] , int row , int col ) {
if ( row == N - 1 && col == N ) return true ;
if ( col == N ) { row ++ ; col = 0 ; }
if ( grid [ row ] [ col ] != 0 ) return solveSuduko ( grid , row , col + 1 ) ; for ( int num = 1 ; num < 10 ; num ++ ) {
if ( isSafe ( grid , row , col , num ) ) {
grid [ row ] [ col ] = num ;
if ( solveSuduko ( grid , row , col + 1 ) ) return true ; }
grid [ row ] [ col ] = 0 ; } return false ; }
int grid [ ] [ ] = { { 3 , 0 , 6 , 5 , 0 , 8 , 4 , 0 , 0 } , { 5 , 2 , 0 , 0 , 0 , 0 , 0 , 0 , 0 } , { 0 , 8 , 7 , 0 , 0 , 0 , 0 , 3 , 1 } , { 0 , 0 , 3 , 0 , 1 , 0 , 0 , 8 , 0 } , { 9 , 0 , 0 , 8 , 6 , 3 , 0 , 0 , 5 } , { 0 , 5 , 0 , 0 , 9 , 0 , 6 , 0 , 0 } , { 1 , 3 , 0 , 0 , 0 , 0 , 2 , 5 , 0 } , { 0 , 0 , 0 , 0 , 0 , 0 , 0 , 7 , 4 } , { 0 , 0 , 5 , 2 , 0 , 6 , 3 , 0 , 0 } } ; if ( solveSuduko ( grid , 0 , 0 ) ) print ( grid ) ; else System . out . println ( " No ▁ Solution ▁ exists " ) ; } }
import java . io . * ; import java . util . HashSet ; class PairSum {
static void printpairs ( int arr [ ] , int sum ) { HashSet < Integer > s = new HashSet < Integer > ( ) ; for ( int i = 0 ; i < arr . length ; ++ i ) { int temp = sum - arr [ i ] ;
if ( s . contains ( temp ) ) { System . out . println ( " Pair ▁ with ▁ given ▁ sum ▁ " + sum + " ▁ is ▁ ( " + arr [ i ] + " , ▁ " + temp + " ) " ) ; } s . add ( arr [ i ] ) ; } }
public static void main ( String [ ] args ) { int A [ ] = { 1 , 4 , 45 , 6 , 10 , 8 } ; int n = 16 ; printpairs ( A , n ) ; } }
import java . io . * ; class GFG { static int exponentMod ( int A , int B , int C ) {
if ( A == 0 ) return 0 ; if ( B == 0 ) return 1 ;
long y ; if ( B % 2 == 0 ) { y = exponentMod ( A , B / 2 , C ) ; y = ( y * y ) % C ; }
else { y = A % C ; y = ( y * exponentMod ( A , B - 1 , C ) % C ) % C ; } return ( int ) ( ( y + C ) % C ) ; }
public static void main ( String args [ ] ) { int A = 2 , B = 5 , C = 13 ; System . out . println ( " Power ▁ is ▁ " + exponentMod ( A , B , C ) ) ; } }
static int power ( int x , int y ) {
int res = 1 ; while ( y > 0 ) {
if ( ( y & 1 ) != 0 ) res = res * x ;
y = y >> 1 ;
x = x * x ; } return res ; }
public class GFG {
static int eggDrop ( int n , int k ) {
if ( k == 1 k == 0 ) return k ;
if ( n == 1 ) return k ; int min = Integer . MAX_VALUE ; int x , res ;
for ( x = 1 ; x <= k ; x ++ ) { res = Math . max ( eggDrop ( n - 1 , x - 1 ) , eggDrop ( n , k - x ) ) ; if ( res < min ) min = res ; } return min + 1 ; }
public static void main ( String args [ ] ) { int n = 2 , k = 10 ; System . out . print ( " Minimum ▁ number ▁ of ▁ " + " trials ▁ in ▁ worst ▁ case ▁ with ▁ " + n + " ▁ eggs ▁ and ▁ " + k + " ▁ floors ▁ is ▁ " + eggDrop ( n , k ) ) ; } }
class Node { int key ; Node left , right ; public Node ( int item ) { key = item ; left = right = null ; } }
public static void main ( String [ ] args ) { BinaryTree tree = new BinaryTree ( ) ; tree . root = new Node ( 1 ) ; tree . root . left = new Node ( 2 ) ; tree . root . right = new Node ( 3 ) ; tree . root . left . left = new Node ( 4 ) ; } }
class Node { int data ; Node left , right ;
public Node ( int data ) { this . data = data ; left = right = null ; } } class BinaryTree { Node root ;
static int findMax ( Node node ) {
if ( node == null ) return Integer . MIN_VALUE ;
int res = node . data ; int lres = findMax ( node . left ) ; int rres = findMax ( node . right ) ; if ( lres > res ) res = lres ; if ( rres > res ) res = rres ; return res ; }
public static void main ( String args [ ] ) { BinaryTree tree = new BinaryTree ( ) ; tree . root = new Node ( 2 ) ; tree . root . left = new Node ( 7 ) ; tree . root . right = new Node ( 5 ) ; tree . root . left . right = new Node ( 6 ) ; tree . root . left . right . left = new Node ( 1 ) ; tree . root . left . right . right = new Node ( 11 ) ; tree . root . right . right = new Node ( 9 ) ; tree . root . right . right . left = new Node ( 4 ) ;
System . out . println ( " Maximum ▁ element ▁ is ▁ " + tree . findMax ( tree . root ) ) ; } }
static int findMin ( Node node ) { if ( node == null ) return Integer . MAX_VALUE ; int res = node . data ; int lres = findMin ( node . left ) ; int rres = findMin ( node . right ) ; if ( lres < res ) res = lres ; if ( rres < res ) res = rres ; return res ; }
class Node { int data ; Node left , right ; Node ( int item ) { data = item ; right = left = null ; } }
public Node extractLeafList ( Node root ) { if ( root == null ) return null ; if ( root . left == null && root . right == null ) { if ( head == null ) { head = root ; prev = root ; } else { prev . right = root ; root . left = prev ; prev = root ; } return null ; } root . left = extractLeafList ( root . left ) ; root . right = extractLeafList ( root . right ) ; return root ; }
void inorder ( Node node ) { if ( node == null ) return ; inorder ( node . left ) ; System . out . print ( node . data + " ▁ " ) ; inorder ( node . right ) ; }
public void printDLL ( Node head ) { Node last = null ; while ( head != null ) { System . out . print ( head . data + " ▁ " ) ; last = head ; head = head . right ; } }
public static void main ( String args [ ] ) { BinaryTree tree = new BinaryTree ( ) ; tree . root = new Node ( 1 ) ; tree . root . left = new Node ( 2 ) ; tree . root . right = new Node ( 3 ) ; tree . root . left . left = new Node ( 4 ) ; tree . root . left . right = new Node ( 5 ) ; tree . root . right . right = new Node ( 6 ) ; tree . root . left . left . left = new Node ( 7 ) ; tree . root . left . left . right = new Node ( 8 ) ; tree . root . right . right . left = new Node ( 9 ) ; tree . root . right . right . right = new Node ( 10 ) ; System . out . println ( " Inorder ▁ traversal ▁ of ▁ given ▁ tree ▁ is ▁ : ▁ " ) ; tree . inorder ( tree . root ) ; tree . extractLeafList ( tree . root ) ; System . out . println ( " " ) ; System . out . println ( " Extracted ▁ double ▁ link ▁ list ▁ is ▁ : ▁ " ) ; tree . printDLL ( tree . head ) ; System . out . println ( " " ) ; System . out . println ( " Inorder ▁ traversal ▁ of ▁ modified ▁ tree ▁ is ▁ : ▁ " ) ; tree . inorder ( tree . root ) ; } }
import java . util . * ; class Solution { static int count = 0 ;
static class Node { int data ; Node left ; Node right ; }
static Node newNode ( int data ) { Node node = new Node ( ) ; node . data = data ; node . left = null ; node . right = null ; return ( node ) ; }
static void NthInorder ( Node node , int n ) { if ( node == null ) return ; if ( count <= n ) {
NthInorder ( node . left , n ) ; count ++ ;
if ( count == n ) System . out . printf ( " % d ▁ " , node . data ) ;
NthInorder ( node . right , n ) ; } }
public static void main ( String args [ ] ) { Node root = newNode ( 10 ) ; root . left = newNode ( 20 ) ; root . right = newNode ( 30 ) ; root . left . left = newNode ( 40 ) ; root . left . right = newNode ( 50 ) ; int n = 4 ; NthInorder ( root , n ) ; } }
static void quickSort ( int arr [ ] , int low , int high ) { if ( low < high ) {
int pi = partition ( arr , low , high ) ;
quickSort ( arr , low , pi - 1 ) ; quickSort ( arr , pi + 1 , high ) ; } }
import java . io . * ; class GFG {
static long countNumberOfStrings ( String s ) {
int n = s . length ( ) - 1 ;
long count = ( long ) ( Math . pow ( 2 , n ) ) ; return count ; }
public static void main ( String [ ] args ) { String S = " ABCD " ; System . out . println ( countNumberOfStrings ( S ) ) ; } }
class GFG {
static void makeArraySumEqual ( int a [ ] , int N ) {
int count_0 = 0 , count_1 = 0 ;
int odd_sum = 0 , even_sum = 0 ; for ( int i = 0 ; i < N ; i ++ ) {
if ( a [ i ] == 0 ) count_0 ++ ;
else count_1 ++ ;
if ( ( i + 1 ) % 2 == 0 ) even_sum += a [ i ] ; else if ( ( i + 1 ) % 2 > 0 ) odd_sum += a [ i ] ; }
if ( odd_sum == even_sum ) {
for ( int i = 0 ; i < N ; i ++ ) System . out . print ( a [ i ] + " ▁ " ) ; }
else { if ( count_0 >= N / 2 ) {
for ( int i = 0 ; i < count_0 ; i ++ ) System . out . print ( "0 ▁ " ) ; } else {
int is_Odd = count_1 % 2 ;
count_1 -= is_Odd ;
for ( int i = 0 ; i < count_1 ; i ++ ) System . out . print ( "1 ▁ " ) ; } } }
public static void main ( String [ ] args ) {
int arr [ ] = { 1 , 1 , 1 , 0 } ; int N = arr . length ;
makeArraySumEqual ( arr , N ) ; } }
class GFG {
static int countDigitSum ( int N , int K ) {
int l = ( int ) Math . pow ( 10 , N - 1 ) , r = ( int ) Math . pow ( 10 , N ) - 1 ; int count = 0 ; for ( int i = l ; i <= r ; i ++ ) { int num = i ;
int digits [ ] = new int [ N ] ; for ( int j = N - 1 ; j >= 0 ; j -- ) { digits [ j ] = num % 10 ; num /= 10 ; } int sum = 0 , flag = 0 ;
for ( int j = 0 ; j < K ; j ++ ) sum += digits [ j ] ;
for ( int j = K ; j < N ; j ++ ) { if ( sum - digits [ j - K ] + digits [ j ] != sum ) { flag = 1 ; break ; } } if ( flag == 0 ) { count ++ ; } } return count ; }
public static void main ( String [ ] args ) {
int N = 2 , K = 1 ; System . out . print ( countDigitSum ( N , K ) ) ; } }
import java . io . * ; class GFG {
static void findpath ( int N , int a [ ] ) {
if ( a [ 0 ] == 1 ) {
System . out . print ( ( N + 1 ) + " ▁ " ) ; for ( int i = 1 ; i <= N ; i ++ ) System . out . print ( ( i ) + " ▁ " ) ; return ; }
for ( int i = 0 ; i < N - 1 ; i ++ ) { if ( a [ i ] == 0 && a [ i + 1 ] == 1 ) {
for ( int j = 1 ; j <= i ; j ++ ) System . out . print ( ( j ) + " ▁ " ) ; System . out . print ( ( N + 1 ) + " ▁ " ) ; for ( int j = i + 1 ; j <= N ; j ++ ) System . out . print ( ( j ) + " ▁ " ) ; return ; } }
for ( int i = 1 ; i <= N ; i ++ ) System . out . print ( ( i ) + " ▁ " ) ; System . out . print ( ( N + 1 ) + " ▁ " ) ; }
public static void main ( String [ ] args ) {
int N = 3 , arr [ ] = { 0 , 1 , 0 } ;
findpath ( N , arr ) ; } }
class GFG {
static int max ( int a , int b ) { return ( a > b ) ? a : b ; }
static void printknapSack ( int W , int wt [ ] , int val [ ] , int n ) { int i , w ; int K [ ] [ ] = new int [ n + 1 ] [ W + 1 ] ;
for ( i = 0 ; i <= n ; i ++ ) { for ( w = 0 ; w <= W ; w ++ ) { if ( i == 0 w == 0 ) K [ i ] [ w ] = 0 ; else if ( wt [ i - 1 ] <= w ) K [ i ] [ w ] = Math . max ( val [ i - 1 ] + K [ i - 1 ] [ w - wt [ i - 1 ] ] , K [ i - 1 ] [ w ] ) ; else K [ i ] [ w ] = K [ i - 1 ] [ w ] ; } }
int res = K [ n ] [ W ] ; System . out . println ( res ) ; w = W ; for ( i = n ; i > 0 && res > 0 ; i -- ) {
if ( res == K [ i - 1 ] [ w ] ) continue ; else {
System . out . print ( wt [ i - 1 ] + " ▁ " ) ;
res = res - val [ i - 1 ] ; w = w - wt [ i - 1 ] ; } } }
public static void main ( String arg [ ] ) { int val [ ] = { 60 , 100 , 120 } ; int wt [ ] = { 10 , 20 , 30 } ; int W = 50 ; int n = val . length ; printknapSack ( W , wt , val , n ) ; } }
public class GFG {
static int optCost ( int freq [ ] , int i , int j ) {
return 0 ;
if ( j == i ) return freq [ i ] ;
int fsum = sum ( freq , i , j ) ;
int min = Integer . MAX_VALUE ;
for ( int r = i ; r <= j ; ++ r ) { int cost = optCost ( freq , i , r - 1 ) + optCost ( freq , r + 1 , j ) ; if ( cost < min ) min = cost ; }
return min + fsum ; }
static int optimalSearchTree ( int keys [ ] , int freq [ ] , int n ) {
return optCost ( freq , 0 , n - 1 ) ; }
static int sum ( int freq [ ] , int i , int j ) { int s = 0 ; for ( int k = i ; k <= j ; k ++ ) s += freq [ k ] ; return s ; }
public static void main ( String [ ] args ) { int keys [ ] = { 10 , 12 , 20 } ; int freq [ ] = { 34 , 8 , 50 } ; int n = keys . length ; System . out . println ( " Cost ▁ of ▁ Optimal ▁ BST ▁ is ▁ " + optimalSearchTree ( keys , freq , n ) ) ; } }
public class WordWrap { final int MAX = Integer . MAX_VALUE ;
int printSolution ( int p [ ] , int n ) { int k ; if ( p [ n ] == 1 ) k = 1 ; else k = printSolution ( p , p [ n ] - 1 ) + 1 ; System . out . println ( " Line ▁ number " + " ▁ " + k + " : ▁ " + " From ▁ word ▁ no . " + " ▁ " + p [ n ] + " ▁ " + " to " + " ▁ " + n ) ; return k ; }
void solveWordWrap ( int l [ ] , int n , int M ) {
int extras [ ] [ ] = new int [ n + 1 ] [ n + 1 ] ;
int lc [ ] [ ] = new int [ n + 1 ] [ n + 1 ] ;
int c [ ] = new int [ n + 1 ] ;
int p [ ] = new int [ n + 1 ] ;
for ( int i = 1 ; i <= n ; i ++ ) { extras [ i ] [ i ] = M - l [ i - 1 ] ; for ( int j = i + 1 ; j <= n ; j ++ ) extras [ i ] [ j ] = extras [ i ] [ j - 1 ] - l [ j - 1 ] - 1 ; }
for ( int i = 1 ; i <= n ; i ++ ) { for ( int j = i ; j <= n ; j ++ ) { if ( extras [ i ] [ j ] < 0 ) lc [ i ] [ j ] = MAX ; else if ( j == n && extras [ i ] [ j ] >= 0 ) lc [ i ] [ j ] = 0 ; else lc [ i ] [ j ] = extras [ i ] [ j ] * extras [ i ] [ j ] ; } }
c [ 0 ] = 0 ; for ( int j = 1 ; j <= n ; j ++ ) { c [ j ] = MAX ; for ( int i = 1 ; i <= j ; i ++ ) { if ( c [ i - 1 ] != MAX && lc [ i ] [ j ] != MAX && ( c [ i - 1 ] + lc [ i ] [ j ] < c [ j ] ) ) { c [ j ] = c [ i - 1 ] + lc [ i ] [ j ] ; p [ j ] = i ; } } } printSolution ( p , n ) ; }
public static void main ( String args [ ] ) { WordWrap w = new WordWrap ( ) ; int l [ ] = { 3 , 2 , 2 , 5 } ; int n = l . length ; int M = 6 ; w . solveWordWrap ( l , n , M ) ; } }
class EggDrop {
static int max ( int a , int b ) { return ( a > b ) ? a : b ; }
static int eggDrop ( int n , int k ) {
int eggFloor [ ] [ ] = new int [ n + 1 ] [ k + 1 ] ; int res ; int i , j , x ;
for ( i = 1 ; i <= n ; i ++ ) { eggFloor [ i ] [ 1 ] = 1 ; eggFloor [ i ] [ 0 ] = 0 ; }
for ( j = 1 ; j <= k ; j ++ ) eggFloor [ 1 ] [ j ] = j ;
for ( i = 2 ; i <= n ; i ++ ) { for ( j = 2 ; j <= k ; j ++ ) { eggFloor [ i ] [ j ] = Integer . MAX_VALUE ; for ( x = 1 ; x <= j ; x ++ ) { res = 1 + max ( eggFloor [ i - 1 ] [ x - 1 ] , eggFloor [ i ] [ j - x ] ) ; if ( res < eggFloor [ i ] [ j ] ) eggFloor [ i ] [ j ] = res ; } } }
return eggFloor [ n ] [ k ] ; }
public static void main ( String args [ ] ) { int n = 2 , k = 10 ; System . out . println ( " Minimum ▁ number ▁ of ▁ trials ▁ in ▁ worst " + " ▁ case ▁ with ▁ " + n + " ▁ eggs ▁ and ▁ " + k + " ▁ floors ▁ is ▁ " + eggDrop ( n , k ) ) ; } }
class Knapsack {
static int max ( int a , int b ) { return ( a > b ) ? a : b ; }
static int knapSack ( int W , int wt [ ] , int val [ ] , int n ) {
if ( n == 0 W == 0 ) return 0 ;
if ( wt [ n - 1 ] > W ) return knapSack ( W , wt , val , n - 1 ) ;
else return max ( val [ n - 1 ] + knapSack ( W - wt [ n - 1 ] , wt , val , n - 1 ) , knapSack ( W , wt , val , n - 1 ) ) ; }
public static void main ( String args [ ] ) { int val [ ] = new int [ ] { 60 , 100 , 120 } ; int wt [ ] = new int [ ] { 10 , 20 , 30 } ; int W = 50 ; int n = val . length ; System . out . println ( knapSack ( W , wt , val , n ) ) ; } }
class LIS {
static int max_ref ;
static int _lis ( int arr [ ] , int n ) {
if ( n == 1 ) return 1 ;
int res , max_ending_here = 1 ;
for ( int i = 1 ; i < n ; i ++ ) { res = _lis ( arr , i ) ; if ( arr [ i - 1 ] < arr [ n - 1 ] && res + 1 > max_ending_here ) max_ending_here = res + 1 ; }
if ( max_ref < max_ending_here ) max_ref = max_ending_here ;
return max_ending_here ; }
static int lis ( int arr [ ] , int n ) {
max_ref = 1 ;
_lis ( arr , n ) ;
return max_ref ; }
public static void main ( String args [ ] ) { int arr [ ] = { 10 , 22 , 9 , 33 , 21 , 50 , 41 , 60 } ; int n = arr . length ; System . out . println ( " Length ▁ of ▁ lis ▁ is ▁ " + lis ( arr , n ) + "NEW_LINE"); } }
public class Main {
public final static int d = 256 ;
static void search ( String pat , String txt , int q ) { int M = pat . length ( ) ; int N = txt . length ( ) ; int i , j ;
int h = 1 ;
for ( i = 0 ; i < M - 1 ; i ++ ) h = ( h * d ) % q ;
for ( i = 0 ; i < M ; i ++ ) { p = ( d * p + pat . charAt ( i ) ) % q ; t = ( d * t + txt . charAt ( i ) ) % q ; }
for ( i = 0 ; i <= N - M ; i ++ ) {
if ( p == t ) {
for ( j = 0 ; j < M ; j ++ ) { if ( txt . charAt ( i + j ) != pat . charAt ( j ) ) break ; }
if ( j == M ) System . out . println ( " Pattern ▁ found ▁ at ▁ index ▁ " + i ) ; }
if ( i < N - M ) { t = ( d * ( t - txt . charAt ( i ) * h ) + txt . charAt ( i + M ) ) % q ;
if ( t < 0 ) t = ( t + q ) ; } } }
public static void main ( String [ ] args ) { String txt = " GEEKS ▁ FOR ▁ GEEKS " ; String pat = " GEEK " ;
int q = 101 ;
search ( pat , txt , q ) ; } }
class KnightTour { static int N = 8 ;
static boolean isSafe ( int x , int y , int sol [ ] [ ] ) { return ( x >= 0 && x < N && y >= 0 && y < N && sol [ x ] [ y ] == - 1 ) ; }
static void printSolution ( int sol [ ] [ ] ) { for ( int x = 0 ; x < N ; x ++ ) { for ( int y = 0 ; y < N ; y ++ ) System . out . print ( sol [ x ] [ y ] + " ▁ " ) ; System . out . println ( ) ; } }
static boolean solveKT ( ) { int sol [ ] [ ] = new int [ 8 ] [ 8 ] ;
for ( int x = 0 ; x < N ; x ++ ) for ( int y = 0 ; y < N ; y ++ ) sol [ x ] [ y ] = - 1 ;
int xMove [ ] = { 2 , 1 , - 1 , - 2 , - 2 , - 1 , 1 , 2 } ; int yMove [ ] = { 1 , 2 , 2 , 1 , - 1 , - 2 , - 2 , - 1 } ;
sol [ 0 ] [ 0 ] = 0 ;
if ( ! solveKTUtil ( 0 , 0 , 1 , sol , xMove , yMove ) ) { System . out . println ( " Solution ▁ does ▁ not ▁ exist " ) ; return false ; } else printSolution ( sol ) ; return true ; }
static boolean solveKTUtil ( int x , int y , int movei , int sol [ ] [ ] , int xMove [ ] , int yMove [ ] ) { int k , next_x , next_y ; if ( movei == N * N ) return true ;
for ( k = 0 ; k < 8 ; k ++ ) { next_x = x + xMove [ k ] ; next_y = y + yMove [ k ] ; if ( isSafe ( next_x , next_y , sol ) ) { sol [ next_x ] [ next_y ] = movei ; if ( solveKTUtil ( next_x , next_y , movei + 1 , sol , xMove , yMove ) ) return true ; else
sol [ next_x ] [ next_y ] = - 1 ; } } return false ; }
public static void main ( String args [ ] ) {
solveKT ( ) ; } }
public class GFG {
static int V = 4 ;
static void printSolution ( int [ ] color ) { System . out . println ( " Solution ▁ Exists : " + " ▁ Following ▁ are ▁ the ▁ assigned ▁ colors ▁ " ) ; for ( int i = 0 ; i < V ; i ++ ) System . out . print ( " ▁ " + color [ i ] ) ; System . out . println ( ) ; }
static boolean isSafe ( boolean [ ] [ ] graph , int [ ] color ) {
for ( int i = 0 ; i < V ; i ++ ) for ( int j = i + 1 ; j < V ; j ++ ) if ( graph [ i ] [ j ] && color [ j ] == color [ i ] ) return false ; return true ; }
static boolean graphColoring ( boolean [ ] [ ] graph , int m , int i , int [ ] color ) {
if ( i == V ) {
if ( isSafe ( graph , color ) ) {
printSolution ( color ) ; return true ; } return false ; }
for ( int j = 1 ; j <= m ; j ++ ) { color [ i ] = j ;
if ( graphColoring ( graph , m , i + 1 , color ) ) return true ; color [ i ] = 0 ; } return false ; }
public static void main ( String [ ] args ) {
boolean [ ] [ ] graph = { { false , true , true , true } , { true , false , true , false } , { true , true , false , true } , { true , false , true , false } , } ;
int m = 3 ;
int [ ] color = new int [ V ] ; for ( int i = 0 ; i < V ; i ++ ) color [ i ] = 0 ; if ( ! graphColoring ( graph , m , 0 , color ) ) System . out . println ( " Solution ▁ does ▁ not ▁ exist " ) ; } }
class GFG {
static int prevPowerofK ( int n , int k ) { int p = ( int ) ( Math . log ( n ) / Math . log ( k ) ) ; return ( int ) Math . pow ( k , p ) ; }
static int nextPowerOfK ( int n , int k ) { return prevPowerofK ( n , k ) * k ; }
public static void main ( String [ ] args ) { int N = 7 ; int K = 2 ; System . out . print ( prevPowerofK ( N , K ) + " ▁ " ) ; System . out . println ( nextPowerOfK ( N , K ) ) ; } }
class Test {
static int gcd ( int a , int b ) { if ( b == 0 ) return a ; return gcd ( b , a % b ) ; }
public static void main ( String [ ] args ) { int a = 98 , b = 56 ; System . out . println ( " GCD ▁ of ▁ " + a + " ▁ and ▁ " + b + " ▁ is ▁ " + gcd ( a , b ) ) ; } }
class GFG {
static int checkSemiprime ( int num ) { int cnt = 0 ; for ( int i = 2 ; cnt < 2 && i * i <= num ; ++ i ) while ( num % i == 0 ) { num /= i ; ++ cnt ; }
if ( num > 1 ) ++ cnt ;
return cnt == 2 ? 1 : 0 ; }
static void semiprime ( int n ) { if ( checkSemiprime ( n ) != 0 ) System . out . printf ( "TrueNEW_LINE"); else System . out . printf ( "FalseNEW_LINE"); }
public static void main ( String [ ] args ) { int n = 6 ; semiprime ( n ) ; n = 8 ; semiprime ( n ) ; } }
class Node { int data ; Node left , right ; Node ( int item ) { data = item ; left = right ; } } class BinaryTree { Node root ;
void reverseLevelOrder ( Node node ) { int h = height ( node ) ; int i ; for ( i = h ; i >= 1 ; i -- ) { printGivenLevel ( node , i ) ; } }
void printGivenLevel ( Node node , int level ) { if ( node == null ) return ; if ( level == 1 ) System . out . print ( node . data + " ▁ " ) ; else if ( level > 1 ) { printGivenLevel ( node . left , level - 1 ) ; printGivenLevel ( node . right , level - 1 ) ; } }
int height ( Node node ) { if ( node == null ) return 0 ; else {
int lheight = height ( node . left ) ; int rheight = height ( node . right ) ;
if ( lheight > rheight ) return ( lheight + 1 ) ; else return ( rheight + 1 ) ; } }
public static void main ( String args [ ] ) { BinaryTree tree = new BinaryTree ( ) ;
tree . root = new Node ( 1 ) ; tree . root . left = new Node ( 2 ) ; tree . root . right = new Node ( 3 ) ; tree . root . left . left = new Node ( 4 ) ; tree . root . left . right = new Node ( 5 ) ; System . out . println ( " Level ▁ Order ▁ traversal ▁ of ▁ binary ▁ tree ▁ is ▁ : ▁ " ) ; tree . reverseLevelOrder ( tree . root ) ; } }
import java . io . * ; class GFG { static void indexedSequentialSearch ( int arr [ ] , int n , int k ) { int elements [ ] = new int [ 20 ] ; int indices [ ] = new int [ 20 ] ; int temp , i ; int j = 0 , ind = 0 , start = 0 , end = 0 , set = 0 ; for ( i = 0 ; i < n ; i += 3 ) {
elements [ ind ] = arr [ i ] ;
indices [ ind ] = i ; ind ++ ; } if ( k < elements [ 0 ] ) { System . out . println ( " Not ▁ found " ) ; return ; } else { for ( i = 1 ; i <= ind ; i ++ ) if ( k <= elements [ i ] ) { start = indices [ i - 1 ] ; set = 1 ; end = indices [ i ] ; break ; } } if ( set == 0 ) { start = indices [ i - 1 ] ; end = n ; } for ( i = start ; i <= end ; i ++ ) { if ( k == arr [ i ] ) { j = 1 ; break ; } } if ( j == 1 ) System . out . println ( " Found ▁ at ▁ index ▁ " + i ) ; else System . out . println ( " Not ▁ found " ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 6 , 7 , 8 , 9 , 10 } ; int n = arr . length ;
int k = 8 ;
indexedSequentialSearch ( arr , n , k ) ; } }
class Main {
static void printNSE ( int arr [ ] , int n ) { int next , i , j ; for ( i = 0 ; i < n ; i ++ ) { next = - 1 ; for ( j = i + 1 ; j < n ; j ++ ) { if ( arr [ i ] > arr [ j ] ) { next = arr [ j ] ; break ; } } System . out . println ( arr [ i ] + " ▁ - - ▁ " + next ) ; } }
public static void main ( String args [ ] ) { int arr [ ] = { 11 , 13 , 21 , 3 } ; int n = arr . length ; printNSE ( arr , n ) ; } }
import java . util . * ; class GFG { static int M = 3 ; static int N = 3 ;
static boolean bpm ( int table [ ] [ ] , int u , boolean seen [ ] , int matchR [ ] ) {
for ( int v = 0 ; v < N ; v ++ ) {
if ( table [ u ] [ v ] > 0 && ! seen [ v ] ) {
seen [ v ] = true ;
if ( matchR [ v ] < 0 || bpm ( table , matchR [ v ] , seen , matchR ) ) { matchR [ v ] = u ; return true ; } } } return false ; }
static int maxBPM ( int table [ ] [ ] ) {
int [ ] matchR = new int [ N ] ;
Arrays . fill ( matchR , - 1 ) ;
int result = 0 ; for ( int u = 0 ; u < M ; u ++ ) {
boolean [ ] seen = new boolean [ N ] ; Arrays . fill ( seen , false ) ;
if ( bpm ( table , u , seen , matchR ) ) result ++ ; } System . out . println ( " The ▁ number ▁ of ▁ maximum ▁ packets " + " ▁ sent ▁ in ▁ the ▁ time ▁ slot ▁ is ▁ " + result ) ; for ( int x = 0 ; x < N ; x ++ ) if ( matchR [ x ] + 1 != 0 ) System . out . println ( " T " + ( matchR [ x ] + 1 ) + " - > ▁ R " + ( x + 1 ) ) ; return result ; }
public static void main ( String [ ] args ) { int table [ ] [ ] = { { 0 , 2 , 0 } , { 3 , 0 , 1 } , { 2 , 4 , 0 } } ; maxBPM ( table ) ; } }
class GFG { static final int SIZE = 100 ;
static String base64Decoder ( char encoded [ ] , int len_str ) { char [ ] decoded_String ; decoded_String = new char [ SIZE ] ; int i , j , k = 0 ;
int num = 0 ;
int count_bits = 0 ;
for ( i = 0 ; i < len_str ; i += 4 ) { num = 0 ; count_bits = 0 ; for ( j = 0 ; j < 4 ; j ++ ) {
if ( encoded [ i + j ] != ' = ' ) { num = num << 6 ; count_bits += 6 ; }
if ( encoded [ i + j ] >= ' A ' && encoded [ i + j ] <= ' Z ' ) num = num | ( encoded [ i + j ] - ' A ' ) ;
else if ( encoded [ i + j ] >= ' a ' && encoded [ i + j ] <= ' z ' ) num = num | ( encoded [ i + j ] - ' a ' + 26 ) ;
else if ( encoded [ i + j ] >= '0' && encoded [ i + j ] <= '9' ) num = num | ( encoded [ i + j ] - '0' + 52 ) ;
else if ( encoded [ i + j ] == ' + ' ) num = num | 62 ;
else if ( encoded [ i + j ] == ' / ' ) num = num | 63 ;
else { num = num >> 2 ; count_bits -= 2 ; } } while ( count_bits != 0 ) { count_bits -= 8 ;
decoded_String [ k ++ ] = ( char ) ( ( num >> count_bits ) & 255 ) ; } } return String . valueOf ( decoded_String ) ; }
public static void main ( String [ ] args ) { char encoded_String [ ] = " TUVOT04 = " . toCharArray ( ) ; int len_str = encoded_String . length ;
len_str -= 1 ; System . out . printf ( "Encoded String : %sNEW_LINE", String . valueOf ( encoded_String ) ) ; System . out . printf ( "Decoded_String : %sNEW_LINE", base64Decoder ( encoded_String , len_str ) ) ; } }
class GFG { static final int NO_OF_CHARS = 256 ;
static void print ( String [ ] list , String word , int list_size ) {
int [ ] map = new int [ NO_OF_CHARS ] ; int i , j , count , word_size ;
for ( i = 0 ; i < word . length ( ) ; i ++ ) map [ word . charAt ( i ) ] = 1 ;
word_size = word . length ( ) ;
for ( i = 0 ; i < list_size ; i ++ ) { for ( j = 0 , count = 0 ; j < list [ i ] . length ( ) ; j ++ ) { if ( map [ list [ i ] . charAt ( j ) ] > 0 ) { count ++ ;
map [ list [ i ] . charAt ( j ) ] = 0 ; } } if ( count == word_size ) System . out . println ( list [ i ] ) ;
for ( j = 0 ; j < word . length ( ) ; j ++ ) map [ word . charAt ( j ) ] = 1 ; } }
public static void main ( String [ ] args ) { String str = " sun " ; String [ ] list = { " geeksforgeeks " , " unsorted " , " sunday " , " just " , " sss " } ; print ( list , str , 5 ) ; } }
class GFG { static final int NO_OF_CHARS = 256 ; static char count [ ] = new char [ NO_OF_CHARS ] ;
static void getCharCountArray ( String str ) { for ( int i = 0 ; i < str . length ( ) ; i ++ ) count [ str . charAt ( i ) ] ++ ; }
static int firstNonRepeating ( String str ) { getCharCountArray ( str ) ; int index = - 1 , i ; for ( i = 0 ; i < str . length ( ) ; i ++ ) { if ( count [ str . charAt ( i ) ] == 1 ) { index = i ; break ; } } return index ; }
public static void main ( String [ ] args ) { String str = " geeksforgeeks " ; int index = firstNonRepeating ( str ) ; System . out . println ( index == - 1 ? " Either ▁ all ▁ characters ▁ are ▁ repeating ▁ or ▁ string ▁ " + " is ▁ empty " : " First ▁ non - repeating ▁ character ▁ is ▁ " + str . charAt ( index ) ) ; } }
class GFG {
static void divideString ( String str , int n ) { int str_size = str . length ( ) ; int part_size ;
if ( str_size % n != 0 ) { System . out . println ( " Invalid ▁ Input : ▁ String ▁ size " + " is ▁ not ▁ divisible ▁ by ▁ n " ) ; return ; }
part_size = str_size / n ; for ( int i = 0 ; i < str_size ; i ++ ) { if ( i % part_size == 0 ) System . out . println ( ) ; System . out . print ( str . charAt ( i ) ) ; } }
String str = " a _ simple _ divide _ string _ quest " ;
divideString ( str , 4 ) ; } }
import java . io . * ; class GFG {
static void cool_line ( int x1 , int y1 , int x2 , int y2 , int x3 , int y3 ) { if ( ( y3 - y2 ) * ( x2 - x1 ) == ( y2 - y1 ) * ( x3 - x2 ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; }
public static void main ( String [ ] args ) { int a1 = 1 , a2 = 1 , a3 = 0 , b1 = 1 , b2 = 6 , b3 = 9 ; cool_line ( a1 , b1 , a2 , b2 , a3 , b3 ) ; } }
import java . io . * ; import static java . lang . Math . pow ; public class A {
static void bestApproximate ( int x [ ] , int y [ ] ) { int n = x . length ; double m , c , sum_x = 0 , sum_y = 0 , sum_xy = 0 , sum_x2 = 0 ; for ( int i = 0 ; i < n ; i ++ ) { sum_x += x [ i ] ; sum_y += y [ i ] ; sum_xy += x [ i ] * y [ i ] ; sum_x2 += pow ( x [ i ] , 2 ) ; } m = ( n * sum_xy - sum_x * sum_y ) / ( n * sum_x2 - pow ( sum_x , 2 ) ) ; c = ( sum_y - m * sum_x ) / n ; System . out . println ( " m ▁ = ▁ " + m ) ; System . out . println ( " c ▁ = ▁ " + c ) ; }
public static void main ( String args [ ] ) { int x [ ] = { 1 , 2 , 3 , 4 , 5 } ; int y [ ] = { 14 , 27 , 40 , 55 , 68 } ; bestApproximate ( x , y ) ; } }
class GFG {
static int findMinInsertions ( char str [ ] , int l , int h ) {
if ( l > h ) return Integer . MAX_VALUE ; if ( l == h ) return 0 ; if ( l == h - 1 ) return ( str [ l ] == str [ h ] ) ? 0 : 1 ;
return ( str [ l ] == str [ h ] ) ? findMinInsertions ( str , l + 1 , h - 1 ) : ( Integer . min ( findMinInsertions ( str , l , h - 1 ) , findMinInsertions ( str , l + 1 , h ) ) + 1 ) ; }
public static void main ( String args [ ] ) { String str = " geeks " ; System . out . println ( findMinInsertions ( str . toCharArray ( ) , 0 , str . length ( ) - 1 ) ) ; } }
class Node { int data ; Node left , right ; Node ( int item ) { data = item ; left = right = null ; } } class BinaryTree { Node root ; void morrisTraversalPreorder ( ) { morrisTraversalPreorder ( root ) ; }
void morrisTraversalPreorder ( Node node ) { while ( node != null ) {
if ( node . left == null ) { System . out . print ( node . data + " ▁ " ) ; node = node . right ; } else {
Node current = node . left ; while ( current . right != null && current . right != node ) { current = current . right ; }
if ( current . right == node ) { current . right = null ; node = node . right ; }
else { System . out . print ( node . data + " ▁ " ) ; current . right = node ; node = node . left ; } } } } void preorder ( ) { preorder ( root ) ; }
void preorder ( Node node ) { if ( node != null ) { System . out . print ( node . data + " ▁ " ) ; preorder ( node . left ) ; preorder ( node . right ) ; } }
public static void main ( String args [ ] ) { BinaryTree tree = new BinaryTree ( ) ; tree . root = new Node ( 1 ) ; tree . root . left = new Node ( 2 ) ; tree . root . right = new Node ( 3 ) ; tree . root . left . left = new Node ( 4 ) ; tree . root . left . right = new Node ( 5 ) ; tree . root . right . left = new Node ( 6 ) ; tree . root . right . right = new Node ( 7 ) ; tree . root . left . left . left = new Node ( 8 ) ; tree . root . left . left . right = new Node ( 9 ) ; tree . root . left . right . left = new Node ( 10 ) ; tree . root . left . right . right = new Node ( 11 ) ; tree . morrisTraversalPreorder ( ) ; System . out . println ( " " ) ; tree . preorder ( ) ; } }
public void push ( int new_data ) {
Node new_node = new Node ( new_data ) ;
new_node . next = head ;
head = new_node ; }
public void insertAfter ( Node prev_node , int new_data ) {
if ( prev_node == null ) { System . out . println ( " The ▁ given ▁ previous ▁ node ▁ cannot ▁ be ▁ null " ) ; return ; }
Node new_node = new Node ( new_data ) ;
new_node . next = prev_node . next ;
prev_node . next = new_node ; }
static void printNthFromLast ( Node head , int n ) { static int i = 0 ; if ( head == null ) return ; printNthFromLast ( head . next , n ) ; if ( ++ i == n ) System . out . print ( head . data ) ; }
class LinkedList {
class Node { int data ; Node next ; Node ( int d ) { data = d ; next = null ; } }
public void push ( int new_data ) {
Node new_node = new Node ( new_data ) ;
new_node . next = head ;
head = new_node ; } void detectLoop ( ) { Node slow_p = head , fast_p = head ; int flag = 0 ; while ( slow_p != null && fast_p != null && fast_p . next != null ) { slow_p = slow_p . next ; fast_p = fast_p . next . next ; if ( slow_p == fast_p ) { flag = 1 ; break ; } } if ( flag == 1 ) System . out . println ( " Loop ▁ found " ) ; else System . out . println ( " Loop ▁ not ▁ found " ) ; }
public static void main ( String args [ ] ) {
LinkedList llist = new LinkedList ( ) ; llist . push ( 20 ) ; llist . push ( 4 ) ; llist . push ( 15 ) ; llist . push ( 10 ) ;
llist . head . next . next . next . next = llist . head ; llist . detectLoop ( ) ; } }
class LinkedList {
class Node { char data ; Node next ; Node ( char d ) { data = d ; next = null ; } }
boolean isPalindrome ( Node head ) { slow_ptr = head ; fast_ptr = head ; Node prev_of_slow_ptr = head ;
Node midnode = null ;
boolean res = true ; if ( head != null && head . next != null ) {
while ( fast_ptr != null && fast_ptr . next != null ) { fast_ptr = fast_ptr . next . next ;
prev_of_slow_ptr = slow_ptr ; slow_ptr = slow_ptr . next ; }
if ( fast_ptr != null ) { midnode = slow_ptr ; slow_ptr = slow_ptr . next ; }
second_half = slow_ptr ;
prev_of_slow_ptr . next = null ;
reverse ( ) ;
res = compareLists ( head , second_half ) ;
reverse ( ) ; if ( midnode != null ) {
prev_of_slow_ptr . next = midnode ; midnode . next = second_half ; } else prev_of_slow_ptr . next = second_half ; } return res ; }
void reverse ( ) { Node prev = null ; Node current = second_half ; Node next ; while ( current != null ) { next = current . next ; current . next = prev ; prev = current ; current = next ; } second_half = prev ; }
boolean compareLists ( Node head1 , Node head2 ) { Node temp1 = head1 ; Node temp2 = head2 ; while ( temp1 != null && temp2 != null ) { if ( temp1 . data == temp2 . data ) { temp1 = temp1 . next ; temp2 = temp2 . next ; } else return false ; }
if ( temp1 == null && temp2 == null ) return true ;
return false ; }
public void push ( char new_data ) {
Node new_node = new Node ( new_data ) ;
new_node . next = head ;
head = new_node ; }
void printList ( Node ptr ) { while ( ptr != null ) { System . out . print ( ptr . data + " - > " ) ; ptr = ptr . next ; } System . out . println ( " NULL " ) ; }
public static void main ( String [ ] args ) {
LinkedList llist = new LinkedList ( ) ; char str [ ] = { ' a ' , ' b ' , ' a ' , ' c ' , ' a ' , ' b ' , ' a ' } ; String string = new String ( str ) ; for ( int i = 0 ; i < 7 ; i ++ ) { llist . push ( str [ i ] ) ; llist . printList ( llist . head ) ; if ( llist . isPalindrome ( llist . head ) != false ) { System . out . println ( " Is ▁ Palindrome " ) ; System . out . println ( " " ) ; } else { System . out . println ( " Not ▁ Palindrome " ) ; System . out . println ( " " ) ; } } } }
class Node { int data ; Node next ; Node ( int d ) { data = d ; next = null ; } } class LinkedList {
public void swapNodes ( int x , int y ) {
if ( x == y ) return ;
Node prevX = null , currX = head ; while ( currX != null && currX . data != x ) { prevX = currX ; currX = currX . next ; }
Node prevY = null , currY = head ; while ( currY != null && currY . data != y ) { prevY = currY ; currY = currY . next ; }
if ( currX == null currY == null ) return ;
if ( prevX != null ) prevX . next = currY ;
else head = currY ;
if ( prevY != null ) prevY . next = currX ;
else head = currX ;
Node temp = currX . next ; currX . next = currY . next ; currY . next = temp ; }
public void push ( int new_data ) {
Node new_Node = new Node ( new_data ) ;
new_Node . next = head ;
head = new_Node ; }
public void printList ( ) { Node tNode = head ; while ( tNode != null ) { System . out . print ( tNode . data + " ▁ " ) ; tNode = tNode . next ; } }
public static void main ( String [ ] args ) { LinkedList llist = new LinkedList ( ) ;
llist . push ( 7 ) ; llist . push ( 6 ) ; llist . push ( 5 ) ; llist . push ( 4 ) ; llist . push ( 3 ) ; llist . push ( 2 ) ; llist . push ( 1 ) ; System . out . print ( " Linked list before calling swapNodes ( ) "); llist . printList ( ) ; llist . swapNodes ( 4 , 3 ) ; System . out . print ( " Linked list after calling swapNodes ( ) "); llist . printList ( ) ; } }
static void pairWiseSwap ( node head ) {
if ( head != null && head . next != null ) {
swap ( head . data , head . next . data ) ;
pairWiseSwap ( head . next . next ) ; } }
else if ( current . data >= new_node . data ) {
Node tmp = current . data ; current . data = new_node . data ; new_node . data = tmp ; new_node . next = ( head_ref ) . next ; ( head_ref ) . next = new_node ; }
public void push ( int new_data ) {
Node new_Node = new Node ( new_data ) ;
new_Node . next = head ; new_Node . prev = null ;
if ( head != null ) head . prev = new_Node ;
head = new_Node ; }
public void InsertAfter ( Node prev_Node , int new_data ) {
if ( prev_Node == null ) { System . out . println ( " The ▁ given ▁ previous ▁ node ▁ cannot ▁ be ▁ NULL ▁ " ) ; return ; }
Node new_node = new Node ( new_data ) ;
new_node . next = prev_Node . next ;
prev_Node . next = new_node ;
new_node . prev = prev_Node ;
if ( new_node . next != null ) new_node . next . prev = new_node ; }
void append ( int new_data ) {
Node new_node = new Node ( new_data ) ; Node last = head ;
new_node . next = null ;
if ( head == null ) { new_node . prev = null ; head = new_node ; return ; }
while ( last . next != null ) last = last . next ;
last . next = new_node ;
new_node . prev = last ; }
import java . io . * ; import java . util . * ;
class Node { int info ; Node prev , next ; } class GFG { static Node head , tail ;
static void nodeInsetail ( int key ) { Node p = new Node ( ) ; p . info = key ; p . next = null ;
if ( head == null ) { head = p ; tail = p ; head . prev = null ; return ; }
if ( p . info < head . info ) { p . prev = null ; head . prev = p ; p . next = head ; head = p ; return ; }
if ( p . info > tail . info ) { p . prev = tail ; tail . next = p ; tail = p ; return ; }
Node temp = head . next ; while ( temp . info < p . info ) temp = temp . next ;
( temp . prev ) . next = p ; p . prev = temp . prev ; temp . prev = p ; p . next = temp ; }
static void printList ( Node temp ) { while ( temp != null ) { System . out . print ( temp . info + " ▁ " ) ; temp = temp . next ; } }
public static void main ( String args [ ] ) { head = tail = null ; nodeInsetail ( 30 ) ; nodeInsetail ( 50 ) ; nodeInsetail ( 90 ) ; nodeInsetail ( 10 ) ; nodeInsetail ( 40 ) ; nodeInsetail ( 110 ) ; nodeInsetail ( 60 ) ; nodeInsetail ( 95 ) ; nodeInsetail ( 23 ) ; System . out . println ( " Doubly ▁ linked ▁ list ▁ on ▁ printing ▁ from ▁ left ▁ to ▁ right " ) ; printList ( head ) ; } }
static class Node { int data ; Node next ; } ;
static void fun1 ( Node head ) { if ( head == null ) { return ; } fun1 ( head . next ) ; System . out . print ( head . data + " ▁ " ) ; }
static void fun2 ( Node head ) { if ( head == null ) { return ; } System . out . print ( head . data + " ▁ " ) ; if ( head . next != null ) { fun2 ( head . next . next ) ; } System . out . print ( head . data + " ▁ " ) ; }
class GFG {
static class Node { int data ; Node next ; } ;
static void fun1 ( Node head ) { if ( head == null ) { return ; } fun1 ( head . next ) ; System . out . print ( head . data + " ▁ " ) ; }
static void fun2 ( Node start ) { if ( start == null ) { return ; } System . out . print ( start . data + " ▁ " ) ; if ( start . next != null ) { fun2 ( start . next . next ) ; } System . out . print ( start . data + " ▁ " ) ; }
static Node push ( Node head_ref , int new_data ) {
Node new_node = new Node ( ) ;
new_node . data = new_data ;
new_node . next = ( head_ref ) ;
( head_ref ) = new_node ; return head_ref ; }
public static void main ( String [ ] args ) {
Node head = null ;
head = push ( head , 5 ) ; head = push ( head , 4 ) ; head = push ( head , 3 ) ; head = push ( head , 2 ) ; head = push ( head , 1 ) ; System . out . print ( " Output ▁ of ▁ fun1 ( ) ▁ for ▁ " + "list 1->2->3->4->5 NEW_LINE"); fun1 ( head ) ; System . out . print ( " Output of fun2 ( ) for " ▁ + ▁ " list 1 -> 2 -> 3 -> 4 -> 5 "); fun2 ( head ) ; } }
class GfG {
static class Node { int data ; Node next ; } static Node head = null ;
static int printsqrtn ( Node head ) { Node sqrtn = null ; int i = 1 , j = 1 ;
while ( head != null ) {
if ( i == j * j ) {
if ( sqrtn == null ) sqrtn = head ; else sqrtn = sqrtn . next ;
j ++ ; } i ++ ; head = head . next ; }
return sqrtn . data ; } static void print ( Node head ) { while ( head != null ) { System . out . print ( head . data + " ▁ " ) ; head = head . next ; } System . out . println ( ) ; }
static void push ( int new_data ) {
Node new_node = new Node ( ) ;
new_node . data = new_data ;
new_node . next = head ;
head = new_node ; }
public static void main ( String [ ] args ) {
push ( 40 ) ; push ( 30 ) ; push ( 20 ) ; push ( 10 ) ; System . out . print ( " Given ▁ linked ▁ list ▁ is : " ) ; print ( head ) ; System . out . print ( " sqrt ( n ) th ▁ node ▁ is ▁ " + printsqrtn ( head ) ) ; } }
class Node { int data ; Node left , right ; Node ( int d ) { data = d ; left = right = null ; } } class BinaryTree { static Node head ;
Node insert ( Node node , int data ) {
if ( node == null ) { return ( new Node ( data ) ) ; } else {
if ( data <= node . data ) { node . left = insert ( node . left , data ) ; } else { node . right = insert ( node . right , data ) ; }
return node ; } }
int minvalue ( Node node ) { Node current = node ;
while ( current . left != null ) { current = current . left ; } return ( current . data ) ; }
public static void main ( String [ ] args ) { BinaryTree tree = new BinaryTree ( ) ; Node root = null ; root = tree . insert ( root , 4 ) ; tree . insert ( root , 2 ) ; tree . insert ( root , 1 ) ; tree . insert ( root , 3 ) ; tree . insert ( root , 6 ) ; tree . insert ( root , 5 ) ; System . out . println ( " Minimum ▁ value ▁ of ▁ BST ▁ is ▁ " + tree . minvalue ( root ) ) ; } }
class Node { char data ; Node left , right ; Node ( char item ) { data = item ; left = right = null ; } } class BinaryTree { Node root ; static int preIndex = 0 ;
Node buildTree ( char in [ ] , char pre [ ] , int inStrt , int inEnd ) { if ( inStrt > inEnd ) return null ;
Node tNode = new Node ( pre [ preIndex ++ ] ) ;
if ( inStrt == inEnd ) return tNode ;
int inIndex = search ( in , inStrt , inEnd , tNode . data ) ;
tNode . left = buildTree ( in , pre , inStrt , inIndex - 1 ) ; tNode . right = buildTree ( in , pre , inIndex + 1 , inEnd ) ; return tNode ; }
int search ( char arr [ ] , int strt , int end , char value ) { int i ; for ( i = strt ; i <= end ; i ++ ) { if ( arr [ i ] == value ) return i ; } return i ; }
void printInorder ( Node node ) { if ( node == null ) return ;
printInorder ( node . left ) ;
System . out . print ( node . data + " ▁ " ) ;
printInorder ( node . right ) ; }
public static void main ( String args [ ] ) { BinaryTree tree = new BinaryTree ( ) ; char in [ ] = new char [ ] { ' D ' , ' B ' , ' E ' , ' A ' , ' F ' , ' C ' } ; char pre [ ] = new char [ ] { ' A ' , ' B ' , ' D ' , ' E ' , ' C ' , ' F ' } ; int len = in . length ; Node root = tree . buildTree ( in , pre , 0 , len - 1 ) ;
System . out . println ( " Inorder ▁ traversal ▁ of ▁ constructed ▁ tree ▁ is ▁ : ▁ " ) ; tree . printInorder ( root ) ; } }
class Node { int data ; Node left , right ; Node ( int item ) { data = item ; left = right = null ; } } class BinaryTree { Node root ;
static Node lca ( Node root , int n1 , int n2 ) { while ( root != null ) {
if ( root . data > n1 && root . data > n2 ) root = root . left ;
else if ( root . data < n1 && root . data < n2 ) root = root . right ; else break ; } return root ; }
public static void main ( String args [ ] ) {
BinaryTree tree = new BinaryTree ( ) ; tree . root = new Node ( 20 ) ; tree . root . left = new Node ( 8 ) ; tree . root . right = new Node ( 22 ) ; tree . root . left . left = new Node ( 4 ) ; tree . root . left . right = new Node ( 12 ) ; tree . root . left . right . left = new Node ( 10 ) ; tree . root . left . right . right = new Node ( 14 ) ; int n1 = 10 , n2 = 14 ; Node t = tree . lca ( tree . root , n1 , n2 ) ; System . out . println ( " LCA ▁ of ▁ " + n1 + " ▁ and ▁ " + n2 + " ▁ is ▁ " + t . data ) ; n1 = 14 ; n2 = 8 ; t = tree . lca ( tree . root , n1 , n2 ) ; System . out . println ( " LCA ▁ of ▁ " + n1 + " ▁ and ▁ " + n2 + " ▁ is ▁ " + t . data ) ; n1 = 10 ; n2 = 22 ; t = tree . lca ( tree . root , n1 , n2 ) ; System . out . println ( " LCA ▁ of ▁ " + n1 + " ▁ and ▁ " + n2 + " ▁ is ▁ " + t . data ) ; } }
class BinaryTree { boolean hasOnlyOneChild ( int pre [ ] , int size ) { int nextDiff , lastDiff ; for ( int i = 0 ; i < size - 1 ; i ++ ) { nextDiff = pre [ i ] - pre [ i + 1 ] ; lastDiff = pre [ i ] - pre [ size - 1 ] ; if ( nextDiff * lastDiff < 0 ) { return false ; } ; } return true ; }
public static void main ( String [ ] args ) { BinaryTree tree = new BinaryTree ( ) ; int pre [ ] = new int [ ] { 8 , 3 , 5 , 7 , 6 } ; int size = pre . length ; if ( tree . hasOnlyOneChild ( pre , size ) == true ) { System . out . println ( " Yes " ) ; } else { System . out . println ( " No " ) ; } } }
class BinaryTree { boolean hasOnlyOneChild ( int pre [ ] , int size ) {
int min , max ; if ( pre [ size - 1 ] > pre [ size - 2 ] ) { max = pre [ size - 1 ] ; min = pre [ size - 2 ] ; } else { max = pre [ size - 2 ] ; min = pre [ size - 1 ] ; }
for ( int i = size - 3 ; i >= 0 ; i -- ) { if ( pre [ i ] < min ) { min = pre [ i ] ; } else if ( pre [ i ] > max ) { max = pre [ i ] ; } else { return false ; } } return true ; }
public static void main ( String [ ] args ) { BinaryTree tree = new BinaryTree ( ) ; int pre [ ] = new int [ ] { 8 , 3 , 5 , 7 , 6 } ; int size = pre . length ; if ( tree . hasOnlyOneChild ( pre , size ) == true ) { System . out . println ( " Yes " ) ; } else { System . out . println ( " No " ) ; } } }
import java . util . * ; class solution {
static class node { int key ; node left ; node right ; int height ; int count ; }
static int height ( node N ) { if ( N == null ) return 0 ; return N . height ; }
static int max ( int a , int b ) { return ( a > b ) ? a : b ; }
static node newNode ( int key ) { node node = new node ( ) ; node . key = key ; node . left = null ; node . right = null ;
node . height = 1 ; node . count = 1 ; return ( node ) ; }
static node rightRotate ( node y ) { node x = y . left ; node T2 = x . right ;
x . right = y ; y . left = T2 ;
y . height = max ( height ( y . left ) , height ( y . right ) ) + 1 ; x . height = max ( height ( x . left ) , height ( x . right ) ) + 1 ;
return x ; }
static node leftRotate ( node x ) { node y = x . right ; node T2 = y . left ;
y . left = x ; x . right = T2 ;
x . height = max ( height ( x . left ) , height ( x . right ) ) + 1 ; y . height = max ( height ( y . left ) , height ( y . right ) ) + 1 ;
return y ; }
static int getBalance ( node N ) { if ( N == null ) return 0 ; return height ( N . left ) - height ( N . right ) ; } static node insert ( node node , int key ) {
if ( node == null ) return ( newNode ( key ) ) ;
if ( key == node . key ) { ( node . count ) ++ ; return node ; }
if ( key < node . key ) node . left = insert ( node . left , key ) ; else node . right = insert ( node . right , key ) ;
node . height = max ( height ( node . left ) , height ( node . right ) ) + 1 ;
int balance = getBalance ( node ) ;
if ( balance > 1 && key < node . left . key ) return rightRotate ( node ) ;
if ( balance < - 1 && key > node . right . key ) return leftRotate ( node ) ;
if ( balance > 1 && key > node . left . key ) { node . left = leftRotate ( node . left ) ; return rightRotate ( node ) ; }
if ( balance < - 1 && key < node . right . key ) { node . right = rightRotate ( node . right ) ; return leftRotate ( node ) ; }
return node ; }
static node minValueNode ( node node ) { node current = node ;
while ( current . left != null ) current = current . left ; return current ; } static node deleteNode ( node root , int key ) {
if ( root == null ) return root ;
if ( key < root . key ) root . left = deleteNode ( root . left , key ) ;
else if ( key > root . key ) root . right = deleteNode ( root . right , key ) ;
else {
if ( root . count > 1 ) { ( root . count ) -- ; return null ; }
if ( ( root . left == null ) || ( root . right == null ) ) { node temp = root . left != null ? root . left : root . right ;
if ( temp == null ) { temp = root ; root = null ; }
else
root = temp ; } else {
node temp = minValueNode ( root . right ) ;
root . key = temp . key ; root . count = temp . count ; temp . count = 1 ;
root . right = deleteNode ( root . right , temp . key ) ; } }
if ( root == null ) return root ;
root . height = max ( height ( root . left ) , height ( root . right ) ) + 1 ;
int balance = getBalance ( root ) ;
if ( balance > 1 && getBalance ( root . left ) >= 0 ) return rightRotate ( root ) ;
if ( balance > 1 && getBalance ( root . left ) < 0 ) { root . left = leftRotate ( root . left ) ; return rightRotate ( root ) ; }
if ( balance < - 1 && getBalance ( root . right ) <= 0 ) return leftRotate ( root ) ;
if ( balance < - 1 && getBalance ( root . right ) > 0 ) { root . right = rightRotate ( root . right ) ; return leftRotate ( root ) ; } return root ; }
static void preOrder ( node root ) { if ( root != null ) { System . out . printf ( " % d ( % d ) ▁ " , root . key , root . count ) ; preOrder ( root . left ) ; preOrder ( root . right ) ; } }
public static void main ( String args [ ] ) { node root = null ;
root = insert ( root , 9 ) ; root = insert ( root , 5 ) ; root = insert ( root , 10 ) ; root = insert ( root , 5 ) ; root = insert ( root , 9 ) ; root = insert ( root , 7 ) ; root = insert ( root , 17 ) ; System . out . printf ( "Pre order traversal of the constructed AVL tree is NEW_LINE"); preOrder ( root ) ; deleteNode ( root , 9 ) ; System . out . printf ( " Pre order traversal after deletion of 9 "); preOrder ( root ) ; } }
class Node { int data ; Node left , right , parent ; Node ( int d ) { data = d ; left = right = parent = null ; } } class BinaryTree { static Node head ;
Node insert ( Node node , int data ) {
if ( node == null ) { return ( new Node ( data ) ) ; } else { Node temp = null ;
if ( data <= node . data ) { temp = insert ( node . left , data ) ; node . left = temp ; temp . parent = node ; } else { temp = insert ( node . right , data ) ; node . right = temp ; temp . parent = node ; }
return node ; } } Node inOrderSuccessor ( Node root , Node n ) {
if ( n . right != null ) { return minValue ( n . right ) ; }
Node p = n . parent ; while ( p != null && n == p . right ) { n = p ; p = p . parent ; } return p ; }
Node minValue ( Node node ) { Node current = node ;
while ( current . left != null ) { current = current . left ; } return current ; }
public static void main ( String [ ] args ) { BinaryTree tree = new BinaryTree ( ) ; Node root = null , temp = null , suc = null , min = null ; root = tree . insert ( root , 20 ) ; root = tree . insert ( root , 8 ) ; root = tree . insert ( root , 22 ) ; root = tree . insert ( root , 4 ) ; root = tree . insert ( root , 12 ) ; root = tree . insert ( root , 10 ) ; root = tree . insert ( root , 14 ) ; temp = root . left . right . right ; suc = tree . inOrderSuccessor ( root , temp ) ; if ( suc != null ) { System . out . println ( " Inorder ▁ successor ▁ of ▁ " + temp . data + " ▁ is ▁ " + suc . data ) ; } else { System . out . println ( " Inorder ▁ successor ▁ does ▁ not ▁ exist " ) ; } } }
import java . util . * ; class GFG {
static class node { int key ; node left ; node right ; } ; static node head ; static node tail ;
static void convertBSTtoDLL ( node root ) {
if ( root == null ) return ;
if ( root . left != null ) convertBSTtoDLL ( root . left ) ;
root . left = tail ;
if ( tail != null ) ( tail ) . right = root ; else head = root ;
tail = root ;
if ( root . right != null ) convertBSTtoDLL ( root . right ) ; }
static boolean isPresentInDLL ( node head , node tail , int sum ) { while ( head != tail ) { int curr = head . key + tail . key ; if ( curr == sum ) return true ; else if ( curr > sum ) tail = tail . left ; else head = head . right ; } return false ; }
static boolean isTripletPresent ( node root ) {
if ( root == null ) return false ;
head = null ; tail = null ; convertBSTtoDLL ( root ) ;
while ( ( head . right != tail ) && ( head . key < 0 ) ) {
if ( isPresentInDLL ( head . right , tail , - 1 * head . key ) ) return true ; else head = head . right ; }
return false ; }
static node newNode ( int num ) { node temp = new node ( ) ; temp . key = num ; temp . left = temp . right = null ; return temp ; }
static node insert ( node root , int key ) { if ( root == null ) return newNode ( key ) ; if ( root . key > key ) root . left = insert ( root . left , key ) ; else root . right = insert ( root . right , key ) ; return root ; }
public static void main ( String [ ] args ) { node root = null ; root = insert ( root , 6 ) ; root = insert ( root , - 13 ) ; root = insert ( root , 14 ) ; root = insert ( root , - 8 ) ; root = insert ( root , 15 ) ; root = insert ( root , 13 ) ; root = insert ( root , 7 ) ; if ( isTripletPresent ( root ) ) System . out . print ( " Present " ) ; else System . out . print ( " Not ▁ Present " ) ; } }
import java . util . * ; class GFG { static final int MAX_SIZE = 100 ;
static class node { int val ; node left , right ; } ;
static class Stack { int size ; int top ; node [ ] array ; } ;
static Stack createStack ( int size ) { Stack stack = new Stack ( ) ; stack . size = size ; stack . top = - 1 ; stack . array = new node [ stack . size ] ; return stack ; }
static int isFull ( Stack stack ) { return ( stack . top - 1 == stack . size ) ? 1 : 0 ; } static int isEmpty ( Stack stack ) { return stack . top == - 1 ? 1 : 0 ; } static void push ( Stack stack , node node ) { if ( isFull ( stack ) == 1 ) return ; stack . array [ ++ stack . top ] = node ; } static node pop ( Stack stack ) { if ( isEmpty ( stack ) == 1 ) return null ; return stack . array [ stack . top -- ] ; }
static boolean isPairPresent ( node root , int target ) {
Stack s1 = createStack ( MAX_SIZE ) ; Stack s2 = createStack ( MAX_SIZE ) ;
boolean done1 = false , done2 = false ; int val1 = 0 , val2 = 0 ; node curr1 = root , curr2 = root ;
while ( true ) {
while ( done1 == false ) { if ( curr1 != null ) { push ( s1 , curr1 ) ; curr1 = curr1 . left ; } else { if ( isEmpty ( s1 ) == 1 ) done1 = true ; else { curr1 = pop ( s1 ) ; val1 = curr1 . val ; curr1 = curr1 . right ; done1 = true ; } } }
while ( done2 == false ) { if ( curr2 != null ) { push ( s2 , curr2 ) ; curr2 = curr2 . right ; } else { if ( isEmpty ( s2 ) == 1 ) done2 = true ; else { curr2 = pop ( s2 ) ; val2 = curr2 . val ; curr2 = curr2 . left ; done2 = true ; } } }
if ( ( val1 != val2 ) && ( val1 + val2 ) == target ) { System . out . print ( " Pair ▁ Found : ▁ " + val1 + " + ▁ " + val2 + " ▁ = ▁ " + target + "NEW_LINE"); return true ; }
else if ( ( val1 + val2 ) < target ) done1 = false ;
else if ( ( val1 + val2 ) > target ) done2 = false ;
if ( val1 >= val2 ) return false ; } }
static node NewNode ( int val ) { node tmp = new node ( ) ; tmp . val = val ; tmp . right = tmp . left = null ; return tmp ; }
public static void main ( String [ ] args ) {
node root = NewNode ( 15 ) ; root . left = NewNode ( 10 ) ; root . right = NewNode ( 20 ) ; root . left . left = NewNode ( 8 ) ; root . left . right = NewNode ( 12 ) ; root . right . left = NewNode ( 16 ) ; root . right . right = NewNode ( 25 ) ; int target = 33 ; if ( isPairPresent ( root , target ) == false ) System . out . print ( " No such values are found "); } }
class node { node left , right ; int data ;
boolean color ; node ( int data ) { this . data = data ; left = null ; right = null ;
color = true ; } } public class LLRBTREE { private static node root = null ;
node rotateLeft ( node myNode ) { System . out . printf ( "left rotation!!NEW_LINE"); node child = myNode . right ; node childLeft = child . left ; child . left = myNode ; myNode . right = childLeft ; return child ; }
node rotateRight ( node myNode ) { System . out . printf ( "right rotationNEW_LINE"); node child = myNode . left ; node childRight = child . right ; child . right = myNode ; myNode . left = childRight ; return child ; }
boolean isRed ( node myNode ) { if ( myNode == null ) return false ; return ( myNode . color == true ) ; }
void swapColors ( node node1 , node node2 ) { boolean temp = node1 . color ; node1 . color = node2 . color ; node2 . color = temp ; }
node insert ( node myNode , int data ) {
if ( myNode == null ) return new node ( data ) ; if ( data < myNode . data ) myNode . left = insert ( myNode . left , data ) ; else if ( data > myNode . data ) myNode . right = insert ( myNode . right , data ) ; else return myNode ;
if ( isRed ( myNode . right ) && ! isRed ( myNode . left ) ) {
myNode = rotateLeft ( myNode ) ;
swapColors ( myNode , myNode . left ) ; }
if ( isRed ( myNode . left ) && isRed ( myNode . left . left ) ) {
myNode = rotateRight ( myNode ) ; swapColors ( myNode , myNode . right ) ; }
if ( isRed ( myNode . left ) && isRed ( myNode . right ) ) {
myNode . color = ! myNode . color ;
myNode . left . color = false ; myNode . right . color = false ; } return myNode ; }
void inorder ( node node ) { if ( node != null ) { inorder ( node . left ) ; System . out . print ( node . data + " ▁ " ) ; inorder ( node . right ) ; } }
public static void main ( String [ ] args ) {
LLRBTREE node = new LLRBTREE ( ) ; root = node . insert ( root , 10 ) ;
root . color = false ; root = node . insert ( root , 20 ) ; root . color = false ; root = node . insert ( root , 30 ) ; root . color = false ; root = node . insert ( root , 40 ) ; root . color = false ; root = node . insert ( root , 50 ) ; root . color = false ; root = node . insert ( root , 25 ) ; root . color = false ;
node . inorder ( root ) ; } }
Node leftMost ( Node n ) { if ( n == null ) return null ; while ( n . left != null ) n = n . left ; return n ; }
static void inOrder ( Node root ) { Node cur = leftMost ( root ) ; while ( cur != null ) { System . out . printf ( " % d ▁ " , cur . data ) ;
if ( cur . rightThread ) cur = cur . right ;
else
cur = leftmost ( cur . right ) ; } }
public class fullbinarytreepostpre { static int preindex ;
static class node { int data ; node left , right ; public node ( int data ) { this . data = data ; } }
static node constructTreeUtil ( int pre [ ] , int post [ ] , int l , int h , int size ) {
if ( preindex >= size l > h ) return null ;
node root = new node ( pre [ preindex ] ) ; preindex ++ ;
if ( l == h preindex >= size ) return root ; int i ;
for ( i = l ; i <= h ; i ++ ) { if ( post [ i ] == pre [ preindex ] ) break ; }
if ( i <= h ) { root . left = constructTreeUtil ( pre , post , l , i , size ) ; root . right = constructTreeUtil ( pre , post , i + 1 , h , size ) ; } return root ; }
static node constructTree ( int pre [ ] , int post [ ] , int size ) { preindex = 0 ; return constructTreeUtil ( pre , post , 0 , size - 1 , size ) ; }
static void printInorder ( node root ) { if ( root == null ) return ; printInorder ( root . left ) ; System . out . print ( root . data + " ▁ " ) ; printInorder ( root . right ) ; }
public static void main ( String [ ] args ) { int pre [ ] = { 1 , 2 , 4 , 8 , 9 , 5 , 3 , 6 , 7 } ; int post [ ] = { 8 , 9 , 4 , 5 , 2 , 6 , 7 , 3 , 1 } ; int size = pre . length ; node root = constructTree ( pre , post , size ) ; System . out . println ( " Inorder ▁ traversal ▁ of ▁ the ▁ constructed ▁ tree : " ) ; printInorder ( root ) ; } }
class GFG { private static void printSorted ( int [ ] arr , int start , int end ) { if ( start > end ) return ;
printSorted ( arr , start * 2 + 1 , end ) ;
System . out . print ( arr [ start ] + " ▁ " ) ;
printSorted ( arr , start * 2 + 2 , end ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 4 , 2 , 5 , 1 , 3 } ; printSorted ( arr , 0 , arr . length - 1 ) ; } }
class Node { int data ; Node left , right ; Node ( int d ) { data = d ; left = right = null ; } } class BinaryTree { Node root ;
int Ceil ( Node node , int input ) {
if ( node == null ) { return - 1 ; }
if ( node . data == input ) { return node . data ; }
if ( node . data < input ) { return Ceil ( node . right , input ) ; }
int ceil = Ceil ( node . left , input ) ; return ( ceil >= input ) ? ceil : node . data ; }
public static void main ( String [ ] args ) { BinaryTree tree = new BinaryTree ( ) ; tree . root = new Node ( 8 ) ; tree . root . left = new Node ( 4 ) ; tree . root . right = new Node ( 12 ) ; tree . root . left . left = new Node ( 2 ) ; tree . root . left . right = new Node ( 6 ) ; tree . root . right . left = new Node ( 10 ) ; tree . root . right . right = new Node ( 14 ) ; for ( int i = 0 ; i < 16 ; i ++ ) { System . out . println ( i + " ▁ " + tree . Ceil ( tree . root , i ) ) ; } } }
class GFG { static class node { int key ; int count ; node left , right ; } ;
static node newNode ( int item ) { node temp = new node ( ) ; temp . key = item ; temp . left = temp . right = null ; temp . count = 1 ; return temp ; }
static void inorder ( node root ) { if ( root != null ) { inorder ( root . left ) ; System . out . print ( root . key + " ( " + root . count + " ) ▁ " ) ; inorder ( root . right ) ; } }
static node insert ( node node , int key ) {
if ( node == null ) return newNode ( key ) ;
if ( key == node . key ) { ( node . count ) ++ ; return node ; }
if ( key < node . key ) node . left = insert ( node . left , key ) ; else node . right = insert ( node . right , key ) ;
return node ; }
static node minValueNode ( node node ) { node current = node ;
while ( current . left != null ) current = current . left ; return current ; }
static node deleteNode ( node root , int key ) {
if ( root == null ) return root ;
if ( key < root . key ) root . left = deleteNode ( root . left , key ) ;
else if ( key > root . key ) root . right = deleteNode ( root . right , key ) ;
else {
if ( root . count > 1 ) { ( root . count ) -- ; return root ; }
if ( root . left == null ) { node temp = root . right ; root = null ; return temp ; } else if ( root . right == null ) { node temp = root . left ; root = null ; return temp ; }
node temp = minValueNode ( root . right ) ;
root . key = temp . key ;
root . right = deleteNode ( root . right , temp . key ) ; } return root ; }
public static void main ( String [ ] args ) {
node root = null ; root = insert ( root , 12 ) ; root = insert ( root , 10 ) ; root = insert ( root , 20 ) ; root = insert ( root , 9 ) ; root = insert ( root , 11 ) ; root = insert ( root , 10 ) ; root = insert ( root , 12 ) ; root = insert ( root , 12 ) ; System . out . print ( " Inorder ▁ traversal ▁ of ▁ " + " the ▁ given ▁ tree ▁ " + "NEW_LINE"); inorder ( root ) ; System . out . print ( " Delete 20 "); root = deleteNode ( root , 20 ) ; System . out . print ( " Inorder ▁ traversal ▁ of ▁ " + "the modified tree NEW_LINE"); inorder ( root ) ; System . out . print ( " Delete 12 "); root = deleteNode ( root , 12 ) ; System . out . print ( " Inorder ▁ traversal ▁ of ▁ " + "the modified tree NEW_LINE"); inorder ( root ) ; System . out . print ( " Delete 9 "); root = deleteNode ( root , 9 ) ; System . out . print ( " Inorder ▁ traversal ▁ of ▁ " + "the modified tree NEW_LINE"); inorder ( root ) ; } }
class GfG { static class node { int key ; node left , right ; } static node root = null ;
static node newNode ( int item ) { node temp = new node ( ) ; temp . key = item ; temp . left = null ; temp . right = null ; return temp ; }
static void inorder ( node root ) { if ( root != null ) { inorder ( root . left ) ; System . out . print ( root . key + " ▁ " ) ; inorder ( root . right ) ; } }
static node insert ( node node , int key ) {
if ( node == null ) return newNode ( key ) ;
if ( key < node . key ) node . left = insert ( node . left , key ) ; else node . right = insert ( node . right , key ) ;
return node ; }
static node minValueNode ( node Node ) { node current = Node ;
while ( current . left != null ) current = current . left ; return current ; }
static node deleteNode ( node root , int key ) {
if ( root == null ) return root ;
if ( key < root . key ) root . left = deleteNode ( root . left , key ) ;
else if ( key > root . key ) root . right = deleteNode ( root . right , key ) ;
else {
if ( root . left == null ) { node temp = root . right ; return temp ; } else if ( root . right == null ) { node temp = root . left ; return temp ; }
node temp = minValueNode ( root . right ) ;
root . key = temp . key ;
root . right = deleteNode ( root . right , temp . key ) ; } return root ; }
static node changeKey ( node root , int oldVal , int newVal ) {
root = deleteNode ( root , oldVal ) ;
root = insert ( root , newVal ) ;
return root ; }
public static void main ( String [ ] args ) {
root = insert ( root , 50 ) ; root = insert ( root , 30 ) ; root = insert ( root , 20 ) ; root = insert ( root , 40 ) ; root = insert ( root , 70 ) ; root = insert ( root , 60 ) ; root = insert ( root , 80 ) ; System . out . println ( " Inorder ▁ traversal ▁ of ▁ the ▁ given ▁ tree " ) ; inorder ( root ) ; root = changeKey ( root , 40 , 10 ) ;
System . out . println ( " Inorder traversal of the modified tree "); inorder ( root ) ; } }
class Node { int info ; Node left , right ; Node ( int d ) { info = d ; left = right = null ; } } class BinaryTree { static Node head ; static int count ;
Node insert ( Node node , int info ) {
if ( node == null ) { return ( new Node ( info ) ) ; } else {
if ( info <= node . info ) { node . left = insert ( node . left , info ) ; } else { node . right = insert ( node . right , info ) ; }
static int check ( int num ) { int sum = 0 , i = num , sum_of_digits , prod_of_digits ;
if ( num < 10 num > 99 ) return 0 ; else { sum_of_digits = ( i % 10 ) + ( i / 10 ) ; prod_of_digits = ( i % 10 ) * ( i / 10 ) ; sum = sum_of_digits + prod_of_digits ; } if ( sum == num ) return 1 ; else return 0 ; }
static void countSpecialDigit ( Node rt ) { int x ; if ( rt == null ) return ; else { x = check ( rt . info ) ; if ( x == 1 ) count = count + 1 ; countSpecialDigit ( rt . left ) ; countSpecialDigit ( rt . right ) ; } }
public static void main ( String [ ] args ) { BinaryTree tree = new BinaryTree ( ) ; Node root = null ; root = tree . insert ( root , 50 ) ; tree . insert ( root , 29 ) ; tree . insert ( root , 59 ) ; tree . insert ( root , 19 ) ; tree . insert ( root , 53 ) ; tree . insert ( root , 556 ) ; tree . insert ( root , 56 ) ; tree . insert ( root , 94 ) ; tree . insert ( root , 13 ) ;
countSpecialDigit ( root ) ; System . out . println ( count ) ; } }
class Node { int data ; Node left , right ; Node ( int item ) { data = item ; left = right = null ; } } class BinaryTree { Node root ;
Node buildTree ( int inorder [ ] , int start , int end , Node node ) { if ( start > end ) return null ;
int i = max ( inorder , start , end ) ;
node = new Node ( inorder [ i ] ) ;
if ( start == end ) return node ;
node . left = buildTree ( inorder , start , i - 1 , node . left ) ; node . right = buildTree ( inorder , i + 1 , end , node . right ) ; return node ; }
int max ( int arr [ ] , int strt , int end ) { int i , max = arr [ strt ] , maxind = strt ; for ( i = strt + 1 ; i <= end ; i ++ ) { if ( arr [ i ] > max ) { max = arr [ i ] ; maxind = i ; } } return maxind ; }
void printInorder ( Node node ) { if ( node == null ) return ;
printInorder ( node . left ) ;
System . out . print ( node . data + " ▁ " ) ;
printInorder ( node . right ) ; }
public static void main ( String args [ ] ) { BinaryTree tree = new BinaryTree ( ) ;
int inorder [ ] = new int [ ] { 5 , 10 , 40 , 30 , 28 } ; int len = inorder . length ; Node mynode = tree . buildTree ( inorder , 0 , len - 1 , tree . root ) ;
System . out . println ( " Inorder ▁ traversal ▁ of ▁ the ▁ constructed ▁ tree ▁ is ▁ " ) ; tree . printInorder ( mynode ) ; } }
class GFG { static int identity ( int num ) { int row , col ; for ( row = 0 ; row < num ; row ++ ) { for ( col = 0 ; col < num ; col ++ ) {
if ( row == col ) System . out . print ( 1 + " ▁ " ) ; else System . out . print ( 0 + " ▁ " ) ; } System . out . println ( ) ; } return 0 ; }
public static void main ( String args [ ] ) { int size = 5 ; identity ( size ) ; } }
class GFG {
private static void search ( int [ ] [ ] mat , int n , int x ) {
int i = 0 , j = n - 1 ; while ( i < n && j >= 0 ) { if ( mat [ i ] [ j ] == x ) { System . out . print ( " n ▁ Found ▁ at ▁ " + i + " ▁ " + j ) ; return ; } if ( mat [ i ] [ j ] > x ) j -- ;
else i ++ ; } System . out . print ( " n ▁ Element ▁ not ▁ found " ) ;
return ; }
public static void main ( String [ ] args ) { int mat [ ] [ ] = { { 10 , 20 , 30 , 40 } , { 15 , 25 , 35 , 45 } , { 27 , 29 , 37 , 48 } , { 32 , 33 , 39 , 50 } } ; search ( mat , 4 , 29 ) ; } }
import java . io . * ; class GFG {
static void fill0X ( int m , int n ) {
int i , k = 0 , l = 0 ;
int r = m , c = n ;
char a [ ] [ ] = new char [ m ] [ n ] ;
char x = ' X ' ;
while ( k < m && l < n ) {
for ( i = l ; i < n ; ++ i ) a [ k ] [ i ] = x ; k ++ ;
for ( i = k ; i < m ; ++ i ) a [ i ] [ n - 1 ] = x ; n -- ;
if ( k < m ) { for ( i = n - 1 ; i >= l ; -- i ) a [ m - 1 ] [ i ] = x ; m -- ; }
if ( l < n ) { for ( i = m - 1 ; i >= k ; -- i ) a [ i ] [ l ] = x ; l ++ ; }
x = ( x == '0' ) ? ' X ' : '0' ; }
for ( i = 0 ; i < r ; i ++ ) { for ( int j = 0 ; j < c ; j ++ ) System . out . print ( a [ i ] [ j ] + " ▁ " ) ; System . out . println ( ) ; } }
public static void main ( String [ ] args ) { System . out . println ( " Output ▁ for ▁ m ▁ = ▁ 5 , ▁ n ▁ = ▁ 6" ) ; fill0X ( 5 , 6 ) ; System . out . println ( " Output ▁ for ▁ m ▁ = ▁ 4 , ▁ n ▁ = ▁ 4" ) ; fill0X ( 4 , 4 ) ; System . out . println ( " Output ▁ for ▁ m ▁ = ▁ 3 , ▁ n ▁ = ▁ 4" ) ; fill0X ( 3 , 4 ) ; } }
import java . io . * ; class GFG { public static int N = 3 ;
static void interchangeDiagonals ( int array [ ] [ ] ) {
for ( int i = 0 ; i < N ; ++ i ) if ( i != N / 2 ) { int temp = array [ i ] [ i ] ; array [ i ] [ i ] = array [ i ] [ N - i - 1 ] ; array [ i ] [ N - i - 1 ] = temp ; } for ( int i = 0 ; i < N ; ++ i ) { for ( int j = 0 ; j < N ; ++ j ) System . out . print ( array [ i ] [ j ] + " ▁ " ) ; System . out . println ( ) ; } }
public static void main ( String [ ] args ) { int array [ ] [ ] = { { 4 , 5 , 6 } , { 1 , 2 , 3 } , { 7 , 8 , 9 } } ; interchangeDiagonals ( array ) ; } }
class Node { int data ; Node left , right ; Node ( int item ) { data = item ; left = right = null ; } } class BinaryTree { Node root ;
Node bintree2listUtil ( Node node ) {
if ( node == null ) return node ;
if ( node . left != null ) {
Node left = bintree2listUtil ( node . left ) ;
for ( ; left . right != null ; left = left . right ) ;
left . right = node ;
node . left = left ; }
if ( node . right != null ) {
Node right = bintree2listUtil ( node . right ) ;
for ( ; right . left != null ; right = right . left ) ;
right . left = node ;
node . right = right ; } return node ; }
Node bintree2list ( Node node ) {
if ( node == null ) return node ;
node = bintree2listUtil ( node ) ;
while ( node . left != null ) node = node . left ; return node ; }
void printList ( Node node ) { while ( node != null ) { System . out . print ( node . data + " ▁ " ) ; node = node . right ; } }
public static void main ( String [ ] args ) { BinaryTree tree = new BinaryTree ( ) ;
tree . root = new Node ( 10 ) ; tree . root . left = new Node ( 12 ) ; tree . root . right = new Node ( 15 ) ; tree . root . left . left = new Node ( 25 ) ; tree . root . left . right = new Node ( 30 ) ; tree . root . right . left = new Node ( 36 ) ;
Node head = tree . bintree2list ( tree . root ) ;
tree . printList ( head ) ; } }
class GFG {
static final int M = 4 ; static final int N = 5 ;
static int findCommon ( int mat [ ] [ ] ) {
int column [ ] = new int [ M ] ;
int min_row ;
int i ; for ( i = 0 ; i < M ; i ++ ) column [ i ] = N - 1 ;
min_row = 0 ;
while ( column [ min_row ] >= 0 ) {
for ( i = 0 ; i < M ; i ++ ) { if ( mat [ i ] [ column [ i ] ] < mat [ min_row ] [ column [ min_row ] ] ) min_row = i ; }
int eq_count = 0 ;
for ( i = 0 ; i < M ; i ++ ) {
if ( mat [ i ] [ column [ i ] ] > mat [ min_row ] [ column [ min_row ] ] ) { if ( column [ i ] == 0 ) return - 1 ;
column [ i ] -= 1 ; } else eq_count ++ ; }
if ( eq_count == M ) return mat [ min_row ] [ column [ min_row ] ] ; } return - 1 ; }
public static void main ( String [ ] args ) { int mat [ ] [ ] = { { 1 , 2 , 3 , 4 , 5 } , { 2 , 4 , 5 , 8 , 10 } , { 3 , 5 , 7 , 9 , 11 } , { 1 , 3 , 5 , 7 , 9 } } ; int result = findCommon ( mat ) ; if ( result == - 1 ) System . out . print ( " No ▁ common ▁ element " ) ; else System . out . print ( " Common ▁ element ▁ is ▁ " + result ) ; } }
public class BinaryTreeToDLL {
static class node { int data ; node left , right ; public node ( int data ) { this . data = data ; } } static node prev ;
static void inorder ( node root ) { if ( root == null ) return ; inorder ( root . left ) ; System . out . print ( root . data + " ▁ " ) ; inorder ( root . right ) ; }
static void fixPrevptr ( node root ) { if ( root == null ) return ; fixPrevptr ( root . left ) ; root . left = prev ; prev = root ; fixPrevptr ( root . right ) ; }
static node fixNextptr ( node root ) {
while ( root . right != null ) root = root . right ;
while ( root != null && root . left != null ) { node left = root . left ; left . right = root ; root = root . left ; }
return root ; }
static node BTTtoDLL ( node root ) { prev = null ;
fixPrevptr ( root ) ;
return fixNextptr ( root ) ; }
static void printlist ( node root ) { while ( root != null ) { System . out . print ( root . data + " ▁ " ) ; root = root . right ; } }
public static void main ( String [ ] args ) {
node root = new node ( 10 ) ; root . left = new node ( 12 ) ; root . right = new node ( 15 ) ; root . left . left = new node ( 25 ) ; root . left . right = new node ( 30 ) ; root . right . left = new node ( 36 ) ; System . out . println ( " Inorder ▁ Tree ▁ Traversal " ) ; inorder ( root ) ; node head = BTTtoDLL ( root ) ; System . out . println ( " DLL Traversal "); printlist ( head ) ; } }
class Node { int data ; Node left , right ; Node ( int item ) { data = item ; left = right = null ; } } class BinaryTree { Node root ;
void convertTree ( Node node ) { int left_data = 0 , right_data = 0 , diff ;
if ( node == null || ( node . left == null && node . right == null ) ) return ; else {
convertTree ( node . left ) ; convertTree ( node . right ) ;
if ( node . left != null ) left_data = node . left . data ;
if ( node . right != null ) right_data = node . right . data ;
diff = left_data + right_data - node . data ;
if ( diff > 0 ) node . data = node . data + diff ;
if ( diff < 0 )
increment ( node , - diff ) ; } }
void increment ( Node node , int diff ) {
if ( node . left != null ) { node . left . data = node . left . data + diff ;
increment ( node . left , diff ) ; }
else if ( node . right != null ) { node . right . data = node . right . data + diff ;
increment ( node . right , diff ) ; } }
void printInorder ( Node node ) { if ( node == null ) return ;
printInorder ( node . left ) ;
System . out . print ( node . data + " ▁ " ) ;
printInorder ( node . right ) ; }
public static void main ( String args [ ] ) { BinaryTree tree = new BinaryTree ( ) ; tree . root = new Node ( 50 ) ; tree . root . left = new Node ( 7 ) ; tree . root . right = new Node ( 2 ) ; tree . root . left . left = new Node ( 3 ) ; tree . root . left . right = new Node ( 5 ) ; tree . root . right . left = new Node ( 1 ) ; tree . root . right . right = new Node ( 30 ) ; System . out . println ( " Inorder ▁ traversal ▁ before ▁ conversion ▁ is ▁ : " ) ; tree . printInorder ( tree . root ) ; tree . convertTree ( tree . root ) ; System . out . println ( " " ) ; System . out . println ( " Inorder ▁ traversal ▁ after ▁ conversion ▁ is ▁ : " ) ; tree . printInorder ( tree . root ) ; } }
class Node { int data ; Node left , right ; Node ( int item ) { data = item ; left = right = null ; } } class BinaryTree { Node root ;
int toSumTree ( Node node ) {
if ( node == null ) return 0 ;
int old_val = node . data ;
node . data = toSumTree ( node . left ) + toSumTree ( node . right ) ;
return node . data + old_val ; }
void printInorder ( Node node ) { if ( node == null ) return ; printInorder ( node . left ) ; System . out . print ( node . data + " ▁ " ) ; printInorder ( node . right ) ; }
public static void main ( String args [ ] ) { BinaryTree tree = new BinaryTree ( ) ;
tree . root = new Node ( 10 ) ; tree . root . left = new Node ( - 2 ) ; tree . root . right = new Node ( 6 ) ; tree . root . left . left = new Node ( 8 ) ; tree . root . left . right = new Node ( - 4 ) ; tree . root . right . left = new Node ( 7 ) ; tree . root . right . right = new Node ( 5 ) ; tree . toSumTree ( tree . root ) ;
System . out . println ( " Inorder ▁ Traversal ▁ of ▁ the ▁ resultant ▁ tree ▁ is : " ) ; tree . printInorder ( tree . root ) ; } }
import java . util . * ; import java . lang . * ; import java . io . * ; class PeakElement {
static int findPeakUtil ( int arr [ ] , int low , int high , int n ) {
int mid = low + ( high - low ) / 2 ;
if ( ( mid == 0 arr [ mid - 1 ] <= arr [ mid ] ) && ( mid == n - 1 arr [ mid + 1 ] <= arr [ mid ] ) ) return mid ;
else if ( mid > 0 && arr [ mid - 1 ] > arr [ mid ] ) return findPeakUtil ( arr , low , ( mid - 1 ) , n ) ;
else return findPeakUtil ( arr , ( mid + 1 ) , high , n ) ; }
static int findPeak ( int arr [ ] , int n ) { return findPeakUtil ( arr , 0 , n - 1 , n ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 3 , 20 , 4 , 1 , 0 } ; int n = arr . length ; System . out . println ( " Index ▁ of ▁ a ▁ peak ▁ point ▁ is ▁ " + findPeak ( arr , n ) ) ; } }
class RepeatElement {
void printRepeating ( int arr [ ] , int size ) { int i , j ; System . out . println ( " Repeated ▁ Elements ▁ are ▁ : " ) ; for ( i = 0 ; i < size ; i ++ ) { for ( j = i + 1 ; j < size ; j ++ ) { if ( arr [ i ] == arr [ j ] ) System . out . print ( arr [ i ] + " ▁ " ) ; } } }
public static void main ( String [ ] args ) { RepeatElement repeat = new RepeatElement ( ) ; int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 1 } ; int arr_size = arr . length ; repeat . printRepeating ( arr , arr_size ) ; } }
class RepeatElement {
void printRepeating ( int arr [ ] , int size ) { int count [ ] = new int [ size ] ; int i ; System . out . println ( " Repeated ▁ elements ▁ are ▁ : ▁ " ) ; for ( i = 0 ; i < size ; i ++ ) { if ( count [ arr [ i ] ] == 1 ) System . out . print ( arr [ i ] + " ▁ " ) ; else count [ arr [ i ] ] ++ ; } }
public static void main ( String [ ] args ) { RepeatElement repeat = new RepeatElement ( ) ; int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 1 } ; int arr_size = arr . length ; repeat . printRepeating ( arr , arr_size ) ; } }
class RepeatElement {
void printRepeating ( int arr [ ] , int size ) {
int S = 0 ;
int P = 1 ;
int x , y ;
int D ; int n = size - 2 , i ;
for ( i = 0 ; i < size ; i ++ ) { S = S + arr [ i ] ; P = P * arr [ i ] ; }
S = S - n * ( n + 1 ) / 2 ;
P = P / fact ( n ) ;
D = ( int ) Math . sqrt ( S * S - 4 * P ) ; x = ( D + S ) / 2 ; y = ( S - D ) / 2 ; System . out . println ( " The ▁ two ▁ repeating ▁ elements ▁ are ▁ : " ) ; System . out . print ( x + " ▁ " + y ) ; }
int fact ( int n ) { return ( n == 0 ) ? 1 : n * fact ( n - 1 ) ; }
public static void main ( String [ ] args ) { RepeatElement repeat = new RepeatElement ( ) ; int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 1 } ; int arr_size = arr . length ; repeat . printRepeating ( arr , arr_size ) ; } }
class RepeatElement { void printRepeating ( int arr [ ] , int size ) {
int xor = arr [ 0 ] ;
int set_bit_no ; int i ; int n = size - 2 ; int x = 0 , y = 0 ;
for ( i = 1 ; i < size ; i ++ ) xor ^= arr [ i ] ; for ( i = 1 ; i <= n ; i ++ ) xor ^= i ;
set_bit_no = ( xor & ~ ( xor - 1 ) ) ;
for ( i = 0 ; i < size ; i ++ ) { int a = arr [ i ] & set_bit_no ; if ( a != 0 ) x = x ^ arr [ i ] ;
else y = y ^ arr [ i ] ;
} for ( i = 1 ; i <= n ; i ++ ) { int a = i & set_bit_no ; if ( a != 0 ) x = x ^ i ;
else y = y ^ i ;
} System . out . println ( " The ▁ two ▁ reppeated ▁ elements ▁ are ▁ : " ) ; System . out . println ( x + " ▁ " + y ) ; }
public static void main ( String [ ] args ) { RepeatElement repeat = new RepeatElement ( ) ; int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 1 } ; int arr_size = arr . length ; repeat . printRepeating ( arr , arr_size ) ; } }
class RepeatElement {
void printRepeating ( int arr [ ] , int size ) { int i ; System . out . println ( " The ▁ repeating ▁ elements ▁ are ▁ : ▁ " ) ; for ( i = 0 ; i < size ; i ++ ) { if ( arr [ Math . abs ( arr [ i ] ) ] > 0 ) arr [ Math . abs ( arr [ i ] ) ] = - arr [ Math . abs ( arr [ i ] ) ] ; else System . out . print ( Math . abs ( arr [ i ] ) + " ▁ " ) ; } }
public static void main ( String [ ] args ) { RepeatElement repeat = new RepeatElement ( ) ; int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 1 } ; int arr_size = arr . length ; repeat . printRepeating ( arr , arr_size ) ; } }
class SubarraySum {
int subArraySum ( int arr [ ] , int n , int sum ) { int curr_sum , i , j ;
for ( i = 0 ; i < n ; i ++ ) { curr_sum = arr [ i ] ;
for ( j = i + 1 ; j <= n ; j ++ ) { if ( curr_sum == sum ) { int p = j - 1 ; System . out . println ( " Sum ▁ found ▁ between ▁ indexes ▁ " + i + " ▁ and ▁ " + p ) ; return 1 ; } if ( curr_sum > sum j == n ) break ; curr_sum = curr_sum + arr [ j ] ; } } System . out . println ( " No ▁ subarray ▁ found " ) ; return 0 ; }
public static void main ( String [ ] args ) { SubarraySum arraysum = new SubarraySum ( ) ; int arr [ ] = { 15 , 2 , 4 , 8 , 9 , 5 , 10 , 23 } ; int n = arr . length ; int sum = 23 ; arraysum . subArraySum ( arr , n , sum ) ; } }
class SubarraySum {
int subArraySum ( int arr [ ] , int n , int sum ) {
int curr_sum = arr [ 0 ] , start = 0 , i ;
for ( i = 1 ; i <= n ; i ++ ) {
while ( curr_sum > sum && start < i - 1 ) { curr_sum = curr_sum - arr [ start ] ; start ++ ; }
if ( curr_sum == sum ) { int p = i - 1 ; System . out . println ( " Sum ▁ found ▁ between ▁ indexes ▁ " + start + " ▁ and ▁ " + p ) ; return 1 ; }
if ( i < n ) curr_sum = curr_sum + arr [ i ] ; }
System . out . println ( " No ▁ subarray ▁ found " ) ; return 0 ; }
public static void main ( String [ ] args ) { SubarraySum arraysum = new SubarraySum ( ) ; int arr [ ] = { 15 , 2 , 4 , 8 , 9 , 5 , 10 , 23 } ; int n = arr . length ; int sum = 23 ; arraysum . subArraySum ( arr , n , sum ) ; } }
class FindTriplet {
boolean find3Numbers ( int A [ ] , int arr_size , int sum ) { int l , r ;
for ( int i = 0 ; i < arr_size - 2 ; i ++ ) {
for ( int j = i + 1 ; j < arr_size - 1 ; j ++ ) {
for ( int k = j + 1 ; k < arr_size ; k ++ ) { if ( A [ i ] + A [ j ] + A [ k ] == sum ) { System . out . print ( " Triplet ▁ is ▁ " + A [ i ] + " , ▁ " + A [ j ] + " , ▁ " + A [ k ] ) ; return true ; } } } }
return false ; }
public static void main ( String [ ] args ) { FindTriplet triplet = new FindTriplet ( ) ; int A [ ] = { 1 , 4 , 45 , 6 , 10 , 8 } ; int sum = 22 ; int arr_size = A . length ; triplet . find3Numbers ( A , arr_size , sum ) ; } }
public class GFG {
static int search ( int arr [ ] , int n , int x ) { int i ; for ( i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == x ) { return i ; } } return - 1 ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 10 , 30 , 15 } ; int x = 30 ; int n = arr . length ; System . out . printf ( " % d ▁ is ▁ present ▁ at ▁ index ▁ % d " , x , search ( arr , n , x ) ) ; } }
class BinarySearch {
int binarySearch ( int arr [ ] , int l , int r , int x ) { if ( r >= l ) { int mid = l + ( r - l ) / 2 ;
if ( arr [ mid ] == x ) return mid ;
if ( arr [ mid ] > x ) return binarySearch ( arr , l , mid - 1 , x ) ;
return binarySearch ( arr , mid + 1 , r , x ) ; }
return - 1 ; }
public static void main ( String args [ ] ) { BinarySearch ob = new BinarySearch ( ) ; int arr [ ] = { 2 , 3 , 4 , 10 , 40 } ; int n = arr . length ; int x = 10 ; int result = ob . binarySearch ( arr , 0 , n - 1 , x ) ; if ( result == - 1 ) System . out . println ( " Element ▁ not ▁ present " ) ; else System . out . println ( " Element ▁ found ▁ at ▁ index ▁ " + result ) ; } }
class BinarySearch {
int binarySearch ( int arr [ ] , int x ) { int l = 0 , r = arr . length - 1 ; while ( l <= r ) { int m = l + ( r - l ) / 2 ;
if ( arr [ m ] == x ) return m ;
if ( arr [ m ] < x ) l = m + 1 ;
else r = m - 1 ; }
return - 1 ; }
public static void main ( String args [ ] ) { BinarySearch ob = new BinarySearch ( ) ; int arr [ ] = { 2 , 3 , 4 , 10 , 40 } ; int n = arr . length ; int x = 10 ; int result = ob . binarySearch ( arr , x ) ; if ( result == - 1 ) System . out . println ( " Element ▁ not ▁ present " ) ; else System . out . println ( " Element ▁ found ▁ at ▁ " + " index ▁ " + result ) ; } }
import java . util . * ; class GFG {
public static int interpolationSearch ( int arr [ ] , int lo , int hi , int x ) { int pos ;
if ( lo <= hi && x >= arr [ lo ] && x <= arr [ hi ] ) {
pos = lo + ( ( ( hi - lo ) / ( arr [ hi ] - arr [ lo ] ) ) * ( x - arr [ lo ] ) ) ;
if ( arr [ pos ] == x ) return pos ;
if ( arr [ pos ] < x ) return interpolationSearch ( arr , pos + 1 , hi , x ) ;
if ( arr [ pos ] > x ) return interpolationSearch ( arr , lo , pos - 1 , x ) ; } return - 1 ; }
public static void main ( String [ ] args ) {
int arr [ ] = { 10 , 12 , 13 , 16 , 18 , 19 , 20 , 21 , 22 , 23 , 24 , 33 , 35 , 42 , 47 } ; int n = arr . length ;
int x = 18 ; int index = interpolationSearch ( arr , 0 , n - 1 , x ) ;
if ( index != - 1 ) System . out . println ( " Element ▁ found ▁ at ▁ index ▁ " + index ) ; else System . out . println ( " Element ▁ not ▁ found . " ) ; } }
class MergeSort {
void merge ( int arr [ ] , int l , int m , int r ) {
int n1 = m - l + 1 ; int n2 = r - m ;
int L [ ] = new int [ n1 ] ; int R [ ] = new int [ n2 ] ;
for ( int i = 0 ; i < n1 ; ++ i ) L [ i ] = arr [ l + i ] ; for ( int j = 0 ; j < n2 ; ++ j ) R [ j ] = arr [ m + 1 + j ] ;
int i = 0 , j = 0 ;
int k = l ; while ( i < n1 && j < n2 ) { if ( L [ i ] <= R [ j ] ) { arr [ k ] = L [ i ] ; i ++ ; } else { arr [ k ] = R [ j ] ; j ++ ; } k ++ ; }
while ( i < n1 ) { arr [ k ] = L [ i ] ; i ++ ; k ++ ; }
while ( j < n2 ) { arr [ k ] = R [ j ] ; j ++ ; k ++ ; } }
void sort ( int arr [ ] , int l , int r ) { if ( l < r ) {
int m = l + ( r - l ) / 2 ;
sort ( arr , l , m ) ; sort ( arr , m + 1 , r ) ;
merge ( arr , l , m , r ) ; } }
static void printArray ( int arr [ ] ) { int n = arr . length ; for ( int i = 0 ; i < n ; ++ i ) System . out . print ( arr [ i ] + " ▁ " ) ; System . out . println ( ) ; }
public static void main ( String args [ ] ) { int arr [ ] = { 12 , 11 , 13 , 5 , 6 , 7 } ; System . out . println ( " Given ▁ Array " ) ; printArray ( arr ) ; MergeSort ob = new MergeSort ( ) ; ob . sort ( arr , 0 , arr . length - 1 ) ; System . out . println ( " Sorted array "); printArray ( arr ) ; } }
import java . util . * ; class QuickSort {
static int partition ( int arr [ ] , int low , int high ) { int pivot = arr [ high ] ; int i = ( low - 1 ) ; for ( int j = low ; j <= high - 1 ; j ++ ) { if ( arr [ j ] <= pivot ) { i ++ ; int temp = arr [ i ] ; arr [ i ] = arr [ j ] ; arr [ j ] = temp ; } } int temp = arr [ i + 1 ] ; arr [ i + 1 ] = arr [ high ] ; arr [ high ] = temp ; return i + 1 ; }
static void quickSortIterative ( int arr [ ] , int l , int h ) {
int [ ] stack = new int [ h - l + 1 ] ;
int top = - 1 ;
stack [ ++ top ] = l ; stack [ ++ top ] = h ;
while ( top >= 0 ) {
h = stack [ top -- ] ; l = stack [ top -- ] ;
int p = partition ( arr , l , h ) ;
if ( p - 1 > l ) { stack [ ++ top ] = l ; stack [ ++ top ] = p - 1 ; }
if ( p + 1 < h ) { stack [ ++ top ] = p + 1 ; stack [ ++ top ] = h ; } } }
public static void main ( String args [ ] ) { int arr [ ] = { 4 , 3 , 5 , 2 , 1 , 3 , 2 , 3 } ; int n = 8 ;
quickSortIterative ( arr , 0 , n - 1 ) ; for ( int i = 0 ; i < n ; i ++ ) { System . out . print ( arr [ i ] + " ▁ " ) ; } } }
import java . util . * ; import java . lang . * ; import java . io . * ; class ActivitySelection {
public static void printMaxActivities ( int s [ ] , int f [ ] , int n ) { int i , j ; System . out . print ( " Following ▁ activities ▁ are ▁ selected ▁ : ▁ n " ) ;
i = 0 ; System . out . print ( i + " ▁ " ) ;
for ( j = 1 ; j < n ; j ++ ) {
if ( s [ j ] >= f [ i ] ) { System . out . print ( j + " ▁ " ) ; i = j ; } } }
public static void main ( String [ ] args ) { int s [ ] = { 1 , 3 , 0 , 5 , 8 , 5 } ; int f [ ] = { 2 , 4 , 6 , 7 , 9 , 9 } ; int n = s . length ; printMaxActivities ( s , f , n ) ; } }
public class LongestCommonSubsequence {
int lcs ( char [ ] X , char [ ] Y , int m , int n ) { if ( m == 0 n == 0 ) return 0 ; if ( X [ m - 1 ] == Y [ n - 1 ] ) return 1 + lcs ( X , Y , m - 1 , n - 1 ) ; else return max ( lcs ( X , Y , m , n - 1 ) , lcs ( X , Y , m - 1 , n ) ) ; }
int max ( int a , int b ) { return ( a > b ) ? a : b ; }
public static void main ( String [ ] args ) { LongestCommonSubsequence lcs = new LongestCommonSubsequence ( ) ; String s1 = " AGGTAB " ; String s2 = " GXTXAYB " ; char [ ] X = s1 . toCharArray ( ) ; char [ ] Y = s2 . toCharArray ( ) ; int m = X . length ; int n = Y . length ; System . out . println ( " Length ▁ of ▁ LCS ▁ is " + " ▁ " + lcs . lcs ( X , Y , m , n ) ) ; } }
public class LongestCommonSubsequence {
int lcs ( char [ ] X , char [ ] Y , int m , int n ) { int L [ ] [ ] = new int [ m + 1 ] [ n + 1 ] ;
for ( int i = 0 ; i <= m ; i ++ ) { for ( int j = 0 ; j <= n ; j ++ ) { if ( i == 0 j == 0 ) L [ i ] [ j ] = 0 ; else if ( X [ i - 1 ] == Y [ j - 1 ] ) L [ i ] [ j ] = L [ i - 1 ] [ j - 1 ] + 1 ; else L [ i ] [ j ] = max ( L [ i - 1 ] [ j ] , L [ i ] [ j - 1 ] ) ; } }
return L [ m ] [ n ] ; }
int max ( int a , int b ) { return ( a > b ) ? a : b ; }
public static void main ( String [ ] args ) { LongestCommonSubsequence lcs = new LongestCommonSubsequence ( ) ; String s1 = " AGGTAB " ; String s2 = " GXTXAYB " ; char [ ] X = s1 . toCharArray ( ) ; char [ ] Y = s2 . toCharArray ( ) ; int m = X . length ; int n = Y . length ; System . out . println ( " Length ▁ of ▁ LCS ▁ is " + " ▁ " + lcs . lcs ( X , Y , m , n ) ) ; } }
public class GFG {
static int min ( int x , int y , int z ) { if ( x < y ) return ( x < z ) ? x : z ; else return ( y < z ) ? y : z ; }
static int minCost ( int cost [ ] [ ] , int m , int n ) { if ( n < 0 m < 0 ) return Integer . MAX_VALUE ; else if ( m == 0 && n == 0 ) return cost [ m ] [ n ] ; else return cost [ m ] [ n ] + min ( minCost ( cost , m - 1 , n - 1 ) , minCost ( cost , m - 1 , n ) , minCost ( cost , m , n - 1 ) ) ; }
public static void main ( String args [ ] ) { int cost [ ] [ ] = { { 1 , 2 , 3 } , { 4 , 8 , 2 } , { 1 , 5 , 3 } } ; System . out . print ( minCost ( cost , 2 , 2 ) ) ; } }
import java . util . * ; class MinimumCostPath { private static int minCost ( int cost [ ] [ ] , int m , int n ) { int i , j ;
int tc [ ] [ ] = new int [ m + 1 ] [ n + 1 ] ; tc [ 0 ] [ 0 ] = cost [ 0 ] [ 0 ] ;
for ( i = 1 ; i <= m ; i ++ ) tc [ i ] [ 0 ] = tc [ i - 1 ] [ 0 ] + cost [ i ] [ 0 ] ;
for ( j = 1 ; j <= n ; j ++ ) tc [ 0 ] [ j ] = tc [ 0 ] [ j - 1 ] + cost [ 0 ] [ j ] ;
for ( i = 1 ; i <= m ; i ++ ) for ( j = 1 ; j <= n ; j ++ ) tc [ i ] [ j ] = min ( tc [ i - 1 ] [ j - 1 ] , tc [ i - 1 ] [ j ] , tc [ i ] [ j - 1 ] ) + cost [ i ] [ j ] ; return tc [ m ] [ n ] ; }
private static int min ( int x , int y , int z ) { if ( x < y ) return ( x < z ) ? x : z ; else return ( y < z ) ? y : z ; }
public static void main ( String args [ ] ) { int cost [ ] [ ] = { { 1 , 2 , 3 } , { 4 , 8 , 2 } , { 1 , 5 , 3 } } ; System . out . println ( minCost ( cost , 2 , 2 ) ) ; } }
class Knapsack {
static int max ( int a , int b ) { return ( a > b ) ? a : b ; }
static int knapSack ( int W , int wt [ ] , int val [ ] , int n ) {
if ( n == 0 W == 0 ) return 0 ;
if ( wt [ n - 1 ] > W ) return knapSack ( W , wt , val , n - 1 ) ;
else return max ( val [ n - 1 ] + knapSack ( W - wt [ n - 1 ] , wt , val , n - 1 ) , knapSack ( W , wt , val , n - 1 ) ) ; }
public static void main ( String args [ ] ) { int val [ ] = new int [ ] { 60 , 100 , 120 } ; int wt [ ] = new int [ ] { 10 , 20 , 30 } ; int W = 50 ; int n = val . length ; System . out . println ( knapSack ( W , wt , val , n ) ) ; } }
class Knapsack {
static int max ( int a , int b ) { return ( a > b ) ? a : b ; }
static int knapSack ( int W , int wt [ ] , int val [ ] , int n ) { int i , w ; int K [ ] [ ] = new int [ n + 1 ] [ W + 1 ] ;
for ( i = 0 ; i <= n ; i ++ ) { for ( w = 0 ; w <= W ; w ++ ) { if ( i == 0 w == 0 ) K [ i ] [ w ] = 0 ; else if ( wt [ i - 1 ] <= w ) K [ i ] [ w ] = max ( val [ i - 1 ] + K [ i - 1 ] [ w - wt [ i - 1 ] ] , K [ i - 1 ] [ w ] ) ; else K [ i ] [ w ] = K [ i - 1 ] [ w ] ; } } return K [ n ] [ W ] ; }
public static void main ( String args [ ] ) { int val [ ] = new int [ ] { 60 , 100 , 120 } ; int wt [ ] = new int [ ] { 10 , 20 , 30 } ; int W = 50 ; int n = val . length ; System . out . println ( knapSack ( W , wt , val , n ) ) ; } }
public class GFG {
static int eggDrop ( int n , int k ) {
if ( k == 1 k == 0 ) return k ;
if ( n == 1 ) return k ; int min = Integer . MAX_VALUE ; int x , res ;
for ( x = 1 ; x <= k ; x ++ ) { res = Math . max ( eggDrop ( n - 1 , x - 1 ) , eggDrop ( n , k - x ) ) ; if ( res < min ) min = res ; } return min + 1 ; }
public static void main ( String args [ ] ) { int n = 2 , k = 10 ; System . out . print ( " Minimum ▁ number ▁ of ▁ " + " trials ▁ in ▁ worst ▁ case ▁ with ▁ " + n + " ▁ eggs ▁ and ▁ " + k + " ▁ floors ▁ is ▁ " + eggDrop ( n , k ) ) ; } }
class GFG {
static int max ( int x , int y ) { return ( x > y ) ? x : y ; }
static int lps ( char seq [ ] , int i , int j ) {
if ( i == j ) { return 1 ; }
if ( seq [ i ] == seq [ j ] && i + 1 == j ) { return 2 ; }
if ( seq [ i ] == seq [ j ] ) { return lps ( seq , i + 1 , j - 1 ) + 2 ; }
return max ( lps ( seq , i , j - 1 ) , lps ( seq , i + 1 , j ) ) ; }
public static void main ( String [ ] args ) { String seq = " GEEKSFORGEEKS " ; int n = seq . length ( ) ; System . out . printf ( " The ▁ length ▁ of ▁ the ▁ LPS ▁ is ▁ % d " , lps ( seq . toCharArray ( ) , 0 , n - 1 ) ) ; } }
public class WordWrap { final int MAX = Integer . MAX_VALUE ;
int printSolution ( int p [ ] , int n ) { int k ; if ( p [ n ] == 1 ) k = 1 ; else k = printSolution ( p , p [ n ] - 1 ) + 1 ; System . out . println ( " Line ▁ number " + " ▁ " + k + " : ▁ " + " From ▁ word ▁ no . " + " ▁ " + p [ n ] + " ▁ " + " to " + " ▁ " + n ) ; return k ; }
void solveWordWrap ( int l [ ] , int n , int M ) {
int extras [ ] [ ] = new int [ n + 1 ] [ n + 1 ] ;
int lc [ ] [ ] = new int [ n + 1 ] [ n + 1 ] ;
int c [ ] = new int [ n + 1 ] ;
int p [ ] = new int [ n + 1 ] ;
for ( int i = 1 ; i <= n ; i ++ ) { extras [ i ] [ i ] = M - l [ i - 1 ] ; for ( int j = i + 1 ; j <= n ; j ++ ) extras [ i ] [ j ] = extras [ i ] [ j - 1 ] - l [ j - 1 ] - 1 ; }
for ( int i = 1 ; i <= n ; i ++ ) { for ( int j = i ; j <= n ; j ++ ) { if ( extras [ i ] [ j ] < 0 ) lc [ i ] [ j ] = MAX ; else if ( j == n && extras [ i ] [ j ] >= 0 ) lc [ i ] [ j ] = 0 ; else lc [ i ] [ j ] = extras [ i ] [ j ] * extras [ i ] [ j ] ; } }
c [ 0 ] = 0 ; for ( int j = 1 ; j <= n ; j ++ ) { c [ j ] = MAX ; for ( int i = 1 ; i <= j ; i ++ ) { if ( c [ i - 1 ] != MAX && lc [ i ] [ j ] != MAX && ( c [ i - 1 ] + lc [ i ] [ j ] < c [ j ] ) ) { c [ j ] = c [ i - 1 ] + lc [ i ] [ j ] ; p [ j ] = i ; } } } printSolution ( p , n ) ; }
public static void main ( String args [ ] ) { WordWrap w = new WordWrap ( ) ; int l [ ] = { 3 , 2 , 2 , 5 } ; int n = l . length ; int M = 6 ; w . solveWordWrap ( l , n , M ) ; } }
public class GFG {
static int sum ( int freq [ ] , int i , int j ) { int s = 0 ; for ( int k = i ; k <= j ; k ++ ) s += freq [ k ] ; return s ; }
static int optCost ( int freq [ ] , int i , int j ) {
if ( j < i ) return 0 ;
if ( j == i ) return freq [ i ] ;
int fsum = sum ( freq , i , j ) ;
int min = Integer . MAX_VALUE ;
for ( int r = i ; r <= j ; ++ r ) { int cost = optCost ( freq , i , r - 1 ) + optCost ( freq , r + 1 , j ) ; if ( cost < min ) min = cost ; }
return min + fsum ; }
static int optimalSearchTree ( int keys [ ] , int freq [ ] , int n ) {
return optCost ( freq , 0 , n - 1 ) ; }
public static void main ( String [ ] args ) { int keys [ ] = { 10 , 12 , 20 } ; int freq [ ] = { 34 , 8 , 50 } ; int n = keys . length ; System . out . println ( " Cost ▁ of ▁ Optimal ▁ BST ▁ is ▁ " + optimalSearchTree ( keys , freq , n ) ) ; } }
public class Optimal_BST2 {
static int sum ( int freq [ ] , int i , int j ) { int s = 0 ; for ( int k = i ; k <= j ; k ++ ) { if ( k >= freq . length ) continue ; s += freq [ k ] ; } return s ; }
static int optimalSearchTree ( int keys [ ] , int freq [ ] , int n ) {
int cost [ ] [ ] = new int [ n + 1 ] [ n + 1 ] ;
for ( int i = 0 ; i < n ; i ++ ) cost [ i ] [ i ] = freq [ i ] ;
for ( int L = 2 ; L <= n ; L ++ ) {
for ( int i = 0 ; i <= n - L + 1 ; i ++ ) {
int j = i + L - 1 ; cost [ i ] [ j ] = Integer . MAX_VALUE ;
for ( int r = i ; r <= j ; r ++ ) {
int c = ( ( r > i ) ? cost [ i ] [ r - 1 ] : 0 ) + ( ( r < j ) ? cost [ r + 1 ] [ j ] : 0 ) + sum ( freq , i , j ) ; if ( c < cost [ i ] [ j ] ) cost [ i ] [ j ] = c ; } } } return cost [ 0 ] [ n - 1 ] ; }
public static void main ( String [ ] args ) { int keys [ ] = { 10 , 12 , 20 } ; int freq [ ] = { 34 , 8 , 50 } ; int n = keys . length ; System . out . println ( " Cost ▁ of ▁ Optimal ▁ BST ▁ is ▁ " + optimalSearchTree ( keys , freq , n ) ) ; } }
class GFG {
static int max ( int x , int y ) { return ( x > y ) ? x : y ; }
static class Node { int data ; Node left , right ; } ;
static int LISS ( Node root ) { if ( root == null ) return 0 ;
int size_excl = LISS ( root . left ) + LISS ( root . right ) ;
int size_incl = 1 ; if ( root . left != null ) size_incl += LISS ( root . left . left ) + LISS ( root . left . right ) ; if ( root . right != null ) size_incl += LISS ( root . right . left ) + LISS ( root . right . right ) ;
return max ( size_incl , size_excl ) ; }
static Node newNode ( int data ) { Node temp = new Node ( ) ; temp . data = data ; temp . left = temp . right = null ; return temp ; }
public static void main ( String args [ ] ) {
Node root = newNode ( 20 ) ; root . left = newNode ( 8 ) ; root . left . left = newNode ( 4 ) ; root . left . right = newNode ( 12 ) ; root . left . right . left = newNode ( 10 ) ; root . left . right . right = newNode ( 14 ) ; root . right = newNode ( 22 ) ; root . right . right = newNode ( 25 ) ; System . out . println ( " Size ▁ of ▁ the ▁ Largest " + " ▁ Independent ▁ Set ▁ is ▁ " + LISS ( root ) ) ; } }
class GFG {
static int getCount ( char keypad [ ] [ ] , int n ) { if ( keypad == null n <= 0 ) return 0 ; if ( n == 1 ) return 10 ;
int [ ] odd = new int [ 10 ] ; int [ ] even = new int [ 10 ] ; int i = 0 , j = 0 , useOdd = 0 , totalCount = 0 ; for ( i = 0 ; i <= 9 ; i ++ )
odd [ i ] = 1 ;
for ( j = 2 ; j <= n ; j ++ ) { useOdd = 1 - useOdd ;
if ( useOdd == 1 ) { even [ 0 ] = odd [ 0 ] + odd [ 8 ] ; even [ 1 ] = odd [ 1 ] + odd [ 2 ] + odd [ 4 ] ; even [ 2 ] = odd [ 2 ] + odd [ 1 ] + odd [ 3 ] + odd [ 5 ] ; even [ 3 ] = odd [ 3 ] + odd [ 2 ] + odd [ 6 ] ; even [ 4 ] = odd [ 4 ] + odd [ 1 ] + odd [ 5 ] + odd [ 7 ] ; even [ 5 ] = odd [ 5 ] + odd [ 2 ] + odd [ 4 ] + odd [ 8 ] + odd [ 6 ] ; even [ 6 ] = odd [ 6 ] + odd [ 3 ] + odd [ 5 ] + odd [ 9 ] ; even [ 7 ] = odd [ 7 ] + odd [ 4 ] + odd [ 8 ] ; even [ 8 ] = odd [ 8 ] + odd [ 0 ] + odd [ 5 ] + odd [ 7 ] + odd [ 9 ] ; even [ 9 ] = odd [ 9 ] + odd [ 6 ] + odd [ 8 ] ; } else { odd [ 0 ] = even [ 0 ] + even [ 8 ] ; odd [ 1 ] = even [ 1 ] + even [ 2 ] + even [ 4 ] ; odd [ 2 ] = even [ 2 ] + even [ 1 ] + even [ 3 ] + even [ 5 ] ; odd [ 3 ] = even [ 3 ] + even [ 2 ] + even [ 6 ] ; odd [ 4 ] = even [ 4 ] + even [ 1 ] + even [ 5 ] + even [ 7 ] ; odd [ 5 ] = even [ 5 ] + even [ 2 ] + even [ 4 ] + even [ 8 ] + even [ 6 ] ; odd [ 6 ] = even [ 6 ] + even [ 3 ] + even [ 5 ] + even [ 9 ] ; odd [ 7 ] = even [ 7 ] + even [ 4 ] + even [ 8 ] ; odd [ 8 ] = even [ 8 ] + even [ 0 ] + even [ 5 ] + even [ 7 ] + even [ 9 ] ; odd [ 9 ] = even [ 9 ] + even [ 6 ] + even [ 8 ] ; } }
totalCount = 0 ; if ( useOdd == 1 ) { for ( i = 0 ; i <= 9 ; i ++ ) totalCount += even [ i ] ; } else { for ( i = 0 ; i <= 9 ; i ++ ) totalCount += odd [ i ] ; } return totalCount ; }
public static void main ( String [ ] args ) { char keypad [ ] [ ] = { { '1' , '2' , '3' } , { '4' , '5' , '6' } , { '7' , '8' , '9' } , { ' * ' , '0' , ' # ' } } ; System . out . printf ( "Count for numbers of length %d: %dNEW_LINE", 1, getCount ( keypad , 1 ) ) ; System . out . printf ( "Count for numbers of length %d: %dNEW_LINE", 2, getCount ( keypad , 2 ) ) ; System . out . printf ( "Count for numbers of length %d: %dNEW_LINE", 3, getCount ( keypad , 3 ) ) ; System . out . printf ( "Count for numbers of length %d: %dNEW_LINE", 4, getCount ( keypad , 4 ) ) ; System . out . printf ( "Count for numbers of length %d: %dNEW_LINE", 5, getCount ( keypad , 5 ) ) ; } }
class GFG {
static int min ( int x , int y ) { return ( x < y ) ? x : y ; }
static class node { int data ; int vc ; node left , right ; } ;
static int vCover ( node root ) {
if ( root == null ) return 0 ; if ( root . left == null && root . right == null ) return 0 ;
if ( root . vc != 0 ) return root . vc ;
int size_incl = 1 + vCover ( root . left ) + vCover ( root . right ) ;
int size_excl = 0 ; if ( root . left != null ) size_excl += 1 + vCover ( root . left . left ) + vCover ( root . left . right ) ; if ( root . right != null ) size_excl += 1 + vCover ( root . right . left ) + vCover ( root . right . right ) ;
root . vc = Math . min ( size_incl , size_excl ) ; return root . vc ; }
static node newNode ( int data ) { node temp = new node ( ) ; temp . data = data ; temp . left = temp . right = null ;
temp . vc = 0 ; return temp ; }
public static void main ( String [ ] args ) {
node root = newNode ( 20 ) ; root . left = newNode ( 8 ) ; root . left . left = newNode ( 4 ) ; root . left . right = newNode ( 12 ) ; root . left . right . left = newNode ( 10 ) ; root . left . right . right = newNode ( 14 ) ; root . right = newNode ( 22 ) ; root . right . right = newNode ( 25 ) ; System . out . printf ( " Size ▁ of ▁ the ▁ smallest ▁ vertex " + " cover ▁ is ▁ % d ▁ " , vCover ( root ) ) ; } }
import java . util . Arrays ; class GFG {
static int count ( int n ) {
int table [ ] = new int [ n + 1 ] , i ;
Arrays . fill ( table , 0 ) ;
table [ 0 ] = 1 ;
for ( i = 3 ; i <= n ; i ++ ) table [ i ] += table [ i - 3 ] ; for ( i = 5 ; i <= n ; i ++ ) table [ i ] += table [ i - 5 ] ; for ( i = 10 ; i <= n ; i ++ ) table [ i ] += table [ i - 10 ] ; return table [ n ] ; }
public static void main ( String [ ] args ) { int n = 20 ; System . out . println ( " Count ▁ for ▁ " + n + " ▁ is ▁ " + count ( n ) ) ; n = 13 ; System . out . println ( " Count ▁ for ▁ " + n + " ▁ is ▁ " + count ( n ) ) ; } }
public class NaiveSearch { public static void search ( String txt , String pat ) { int M = pat . length ( ) ; int N = txt . length ( ) ;
for ( int i = 0 ; i <= N - M ; i ++ ) { int j ;
for ( j = 0 ; j < M ; j ++ ) if ( txt . charAt ( i + j ) != pat . charAt ( j ) ) break ;
if ( j == M ) System . out . println ( " Pattern ▁ found ▁ at ▁ index ▁ " + i ) ; } }
public static void main ( String [ ] args ) { String txt = " AABAACAADAABAAABAA " ; String pat = " AABA " ; search ( txt , pat ) ; } }
public class Main {
public final static int d = 256 ;
static void search ( String pat , String txt , int q ) { int M = pat . length ( ) ; int N = txt . length ( ) ; int i , j ;
int p = 0 ;
int t = 0 ; int h = 1 ;
for ( i = 0 ; i < M - 1 ; i ++ ) h = ( h * d ) % q ;
for ( i = 0 ; i < M ; i ++ ) { p = ( d * p + pat . charAt ( i ) ) % q ; t = ( d * t + txt . charAt ( i ) ) % q ; }
for ( i = 0 ; i <= N - M ; i ++ ) {
if ( p == t ) {
for ( j = 0 ; j < M ; j ++ ) { if ( txt . charAt ( i + j ) != pat . charAt ( j ) ) break ; }
if ( j == M ) System . out . println ( " Pattern ▁ found ▁ at ▁ index ▁ " + i ) ; }
if ( i < N - M ) { t = ( d * ( t - txt . charAt ( i ) * h ) + txt . charAt ( i + M ) ) % q ;
if ( t < 0 ) t = ( t + q ) ; } } }
public static void main ( String [ ] args ) { String txt = " GEEKS ▁ FOR ▁ GEEKS " ; String pat = " GEEK " ;
int q = 101 ;
search ( pat , txt , q ) ; } }
class GFG {
static void search ( String pat , String txt ) { int M = pat . length ( ) ; int N = txt . length ( ) ; int i = 0 ; while ( i <= N - M ) { int j ;
for ( j = 0 ; j < M ; j ++ ) if ( txt . charAt ( i + j ) != pat . charAt ( j ) ) break ;
if ( j == M ) { System . out . println ( " Pattern ▁ found ▁ at ▁ index ▁ " + i ) ; i = i + M ; } else if ( j == 0 ) i = i + 1 ; else
i = i + j ; } }
public static void main ( String [ ] args ) { String txt = " ABCEABCDABCEABCD " ; String pat = " ABCD " ; search ( pat , txt ) ; } }
class GFG { static int NO_OF_CHARS = 256 ; static int getNextState ( char [ ] pat , int M , int state , int x ) {
if ( state < M && x == pat [ state ] ) return state + 1 ;
int ns , i ;
for ( ns = state ; ns > 0 ; ns -- ) { if ( pat [ ns - 1 ] == x ) { for ( i = 0 ; i < ns - 1 ; i ++ ) if ( pat [ i ] != pat [ state - ns + 1 + i ] ) break ; if ( i == ns - 1 ) return ns ; } } return 0 ; }
static void computeTF ( char [ ] pat , int M , int TF [ ] [ ] ) { int state , x ; for ( state = 0 ; state <= M ; ++ state ) for ( x = 0 ; x < NO_OF_CHARS ; ++ x ) TF [ state ] [ x ] = getNextState ( pat , M , state , x ) ; }
static void search ( char [ ] pat , char [ ] txt ) { int M = pat . length ; int N = txt . length ; int [ ] [ ] TF = new int [ M + 1 ] [ NO_OF_CHARS ] ; computeTF ( pat , M , TF ) ;
int i , state = 0 ; for ( i = 0 ; i < N ; i ++ ) { state = TF [ state ] [ txt [ i ] ] ; if ( state == M ) System . out . println ( " Pattern ▁ found ▁ " + " at ▁ index ▁ " + ( i - M + 1 ) ) ; } }
public static void main ( String [ ] args ) { char [ ] pat = " AABAACAADAABAAABAA " . toCharArray ( ) ; char [ ] txt = " AABA " . toCharArray ( ) ; search ( txt , pat ) ; } }
class AWQ { static int NO_OF_CHARS = 256 ;
static int max ( int a , int b ) { return ( a > b ) ? a : b ; }
static void badCharHeuristic ( char [ ] str , int size , int badchar [ ] ) {
for ( int i = 0 ; i < NO_OF_CHARS ; i ++ ) badchar [ i ] = - 1 ;
for ( i = 0 ; i < size ; i ++ ) badchar [ ( int ) str [ i ] ] = i ; }
static void search ( char txt [ ] , char pat [ ] ) { int m = pat . length ; int n = txt . length ; int badchar [ ] = new int [ NO_OF_CHARS ] ;
badCharHeuristic ( pat , m , badchar ) ;
int s = 0 ;
while ( s <= ( n - m ) ) { int j = m - 1 ;
while ( j >= 0 && pat [ j ] == txt [ s + j ] ) j -- ;
if ( j < 0 ) { System . out . println ( " Patterns ▁ occur ▁ at ▁ shift ▁ = ▁ " + s ) ;
s += ( s + m < n ) ? m - badchar [ txt [ s + m ] ] : 1 ; } else
s += max ( 1 , j - badchar [ txt [ s + j ] ] ) ; } }
public static void main ( String [ ] args ) { char txt [ ] = " ABAAABCD " . toCharArray ( ) ; char pat [ ] = " ABC " . toCharArray ( ) ; search ( txt , pat ) ; } }
public class Suduko {
static int N = 9 ;
static void print ( int [ ] [ ] grid ) { for ( int i = 0 ; i < N ; i ++ ) { for ( int j = 0 ; j < N ; j ++ ) System . out . print ( grid [ i ] [ j ] + " ▁ " ) ; System . out . println ( ) ; } }
static boolean isSafe ( int [ ] [ ] grid , int row , int col , int num ) {
for ( int x = 0 ; x <= 8 ; x ++ ) if ( grid [ row ] [ x ] == num ) return false ;
for ( int x = 0 ; x <= 8 ; x ++ ) if ( grid [ x ] [ col ] == num ) return false ;
int startRow = row - row % 3 , startCol = col - col % 3 ; for ( int i = 0 ; i < 3 ; i ++ ) for ( int j = 0 ; j < 3 ; j ++ ) if ( grid [ i + startRow ] [ j + startCol ] == num ) return false ; return true ; }
static boolean solveSuduko ( int grid [ ] [ ] , int row , int col ) {
if ( row == N - 1 && col == N ) return true ;
if ( col == N ) { row ++ ; col = 0 ; }
if ( grid [ row ] [ col ] != 0 ) return solveSuduko ( grid , row , col + 1 ) ; for ( int num = 1 ; num < 10 ; num ++ ) {
if ( isSafe ( grid , row , col , num ) ) {
grid [ row ] [ col ] = num ;
if ( solveSuduko ( grid , row , col + 1 ) ) return true ; }
grid [ row ] [ col ] = 0 ; } return false ; }
public static void main ( String [ ] args ) { int grid [ ] [ ] = { { 3 , 0 , 6 , 5 , 0 , 8 , 4 , 0 , 0 } , { 5 , 2 , 0 , 0 , 0 , 0 , 0 , 0 , 0 } , { 0 , 8 , 7 , 0 , 0 , 0 , 0 , 3 , 1 } , { 0 , 0 , 3 , 0 , 1 , 0 , 0 , 8 , 0 } , { 9 , 0 , 0 , 8 , 6 , 3 , 0 , 0 , 5 } , { 0 , 5 , 0 , 0 , 9 , 0 , 6 , 0 , 0 } , { 1 , 3 , 0 , 0 , 0 , 0 , 2 , 5 , 0 } , { 0 , 0 , 0 , 0 , 0 , 0 , 0 , 7 , 4 } , { 0 , 0 , 5 , 2 , 0 , 6 , 3 , 0 , 0 } } ; if ( solveSuduko ( grid , 0 , 0 ) ) print ( grid ) ; else System . out . println ( " No ▁ Solution ▁ exists " ) ; } }
class Main {
static int getMedian ( int ar1 [ ] , int ar2 [ ] , int n ) { int i = 0 ; int j = 0 ; int count ; int m1 = - 1 , m2 = - 1 ;
for ( count = 0 ; count <= n ; count ++ ) {
if ( i == n ) { m1 = m2 ; m2 = ar2 [ 0 ] ; break ; }
else if ( j == n ) { m1 = m2 ; m2 = ar1 [ 0 ] ; break ; }
if ( ar1 [ i ] <= ar2 [ j ] ) {
m1 = m2 ; m2 = ar1 [ i ] ; i ++ ; } else {
m1 = m2 ; m2 = ar2 [ j ] ; j ++ ; } } return ( m1 + m2 ) / 2 ; }
public static void main ( String [ ] args ) { int ar1 [ ] = { 1 , 12 , 15 , 26 , 38 } ; int ar2 [ ] = { 2 , 13 , 17 , 30 , 45 } ; int n1 = ar1 . length ; int n2 = ar2 . length ; if ( n1 == n2 ) System . out . println ( " Median ▁ is ▁ " + getMedian ( ar1 , ar2 , n1 ) ) ; else System . out . println ( " arrays ▁ are ▁ of ▁ unequal ▁ size " ) ; } }
import java . util . * ; class GfG {
static int getMedian ( int [ ] a , int [ ] b , int startA , int startB , int endA , int endB ) { if ( endA - startA == 1 ) { return ( Math . max ( a [ startA ] , b [ startB ] ) + Math . min ( a [ endA ] , b [ endB ] ) ) / 2 ; }
int m1 = median ( a , startA , endA ) ;
int m2 = median ( b , startB , endB ) ;
if ( m1 == m2 ) { return m1 ; }
else if ( m1 < m2 ) { return getMedian ( a , b , ( endA + startA + 1 ) / 2 , startB , endA , ( endB + startB + 1 ) / 2 ) ; }
else { return getMedian ( a , b , startA , ( endB + startB + 1 ) / 2 , ( endA + startA + 1 ) / 2 , endB ) ; } }
static int median ( int [ ] arr , int start , int end ) { int n = end - start + 1 ; if ( n % 2 == 0 ) { return ( arr [ start + ( n / 2 ) ] + arr [ start + ( n / 2 - 1 ) ] ) / 2 ; } else { return arr [ start + n / 2 ] ; } }
public static void main ( String [ ] args ) { int ar1 [ ] = { 1 , 2 , 3 , 6 } ; int ar2 [ ] = { 4 , 6 , 8 , 10 } ; int n1 = ar1 . length ; int n2 = ar2 . length ; if ( n1 != n2 ) { System . out . println ( " Doesn ' t ▁ work ▁ for ▁ arrays ▁ " + " of ▁ unequal ▁ size " ) ; } else if ( n1 == 0 ) { System . out . println ( " Arrays ▁ are ▁ empty . " ) ; } else if ( n1 == 1 ) { System . out . println ( ( ar1 [ 0 ] + ar2 [ 0 ] ) / 2 ) ; } else { System . out . println ( " Median ▁ is ▁ " + getMedian ( ar1 , ar2 , 0 , 0 , ar1 . length - 1 , ar2 . length - 1 ) ) ; } } }
import java . io . * ; class GFG { public static int counter = 2 ;
static boolean isLucky ( int n ) {
int next_position = n ; if ( counter > n ) return true ; if ( n % counter == 0 ) return false ;
next_position -= next_position / counter ; counter ++ ; return isLucky ( next_position ) ; }
public static void main ( String [ ] args ) { int x = 5 ; if ( isLucky ( x ) ) System . out . println ( x + " ▁ is ▁ a ▁ lucky ▁ no . " ) ; else System . out . println ( x + " ▁ is ▁ not ▁ a ▁ lucky ▁ no . " ) ; } }
import java . io . * ; class GFG {
static int pow ( int a , int b ) { if ( b == 0 ) return 1 ; int answer = a ; int increment = a ; int i , j ; for ( i = 1 ; i < b ; i ++ ) { for ( j = 1 ; j < a ; j ++ ) { answer += increment ; } increment = answer ; } return answer ; }
public static void main ( String [ ] args ) { System . out . println ( pow ( 5 , 3 ) ) ; } }
import java . io . * ; class GFG {
static int multiply ( int x , int y ) { if ( y > 0 ) return ( x + multiply ( x , y - 1 ) ) ; else return 0 ; }
static int pow ( int a , int b ) { if ( b > 0 ) return multiply ( a , pow ( a , b - 1 ) ) ; else return 1 ; }
public static void main ( String [ ] args ) { System . out . println ( pow ( 5 , 3 ) ) ; } }
import java . io . * ; class GFG {
static int count ( int n ) {
if ( n < 3 ) return n ; if ( n >= 3 && n < 10 ) return n - 1 ;
int po = 1 ; while ( n / po > 9 ) po = po * 10 ;
int msd = n / po ; if ( msd != 3 )
return count ( msd ) * count ( po - 1 ) + count ( msd ) + count ( n % po ) ; else
return count ( msd * po - 1 ) ; }
public static void main ( String [ ] args ) { int n = 578 ; System . out . println ( count ( n ) ) ; } }
import java . io . * ; import java . util . * ; class GFG {
static int fact ( int n ) { return ( n <= 1 ) ? 1 : n * fact ( n - 1 ) ; }
static int findSmallerInRight ( String str , int low , int high ) { int countRight = 0 , i ; for ( i = low + 1 ; i <= high ; ++ i ) if ( str . charAt ( i ) < str . charAt ( low ) ) ++ countRight ; return countRight ; }
static int findRank ( String str ) { int len = str . length ( ) ; int mul = fact ( len ) ; int rank = 1 ; int countRight ; for ( int i = 0 ; i < len ; ++ i ) { mul /= len - i ;
countRight = findSmallerInRight ( str , i , len - 1 ) ; rank += countRight * mul ; } return rank ; }
public static void main ( String [ ] args ) { String str = " string " ; System . out . println ( findRank ( str ) ) ; } }
class GFG { static int MAX_CHAR = 256 ;
int count [ ] = new int [ MAX_CHAR ] ;
static int fact ( int n ) { return ( n <= 1 ) ? 1 : n * fact ( n - 1 ) ; }
static void populateAndIncreaseCount ( int [ ] count , char [ ] str ) { int i ; for ( i = 0 ; i < str . length ; ++ i ) ++ count [ str [ i ] ] ; for ( i = 1 ; i < MAX_CHAR ; ++ i ) count [ i ] += count [ i - 1 ] ; }
static void updatecount ( int [ ] count , char ch ) { int i ; for ( i = ch ; i < MAX_CHAR ; ++ i ) -- count [ i ] ; }
static int findRank ( char [ ] str ) { int len = str . length ; int mul = fact ( len ) ; int rank = 1 , i ;
populateAndIncreaseCount ( count , str ) ; for ( i = 0 ; i < len ; ++ i ) { mul /= len - i ;
rank += count [ str [ i ] - 1 ] * mul ;
updatecount ( count , str [ i ] ) ; } return rank ; }
public static void main ( String args [ ] ) { char str [ ] = " string " . toCharArray ( ) ; System . out . println ( findRank ( str ) ) ; } }
import java . io . * ; class GFG {
static float exponential ( int n , float x ) {
float sum = 1 ; for ( int i = n - 1 ; i > 0 ; -- i ) sum = 1 + x * sum / i ; return sum ; }
public static void main ( String [ ] args ) { int n = 10 ; float x = 1 ; System . out . println ( " e ^ x ▁ = ▁ " + exponential ( n , x ) ) ; } }
class GFG {
static int findCeil ( int arr [ ] , int r , int l , int h ) { int mid ; while ( l < h ) {
mid = l + ( ( h - l ) >> 1 ) ; if ( r > arr [ mid ] ) l = mid + 1 ; else h = mid ; } return ( arr [ l ] >= r ) ? l : - 1 ; }
static int myRand ( int arr [ ] , int freq [ ] , int n ) {
int prefix [ ] = new int [ n ] , i ; prefix [ 0 ] = freq [ 0 ] ; for ( i = 1 ; i < n ; ++ i ) prefix [ i ] = prefix [ i - 1 ] + freq [ i ] ;
int r = ( ( int ) ( Math . random ( ) * ( 323567 ) ) % prefix [ n - 1 ] ) + 1 ;
int indexc = findCeil ( prefix , r , 0 , n - 1 ) ; return arr [ indexc ] ; }
public static void main ( String args [ ] ) { int arr [ ] = { 1 , 2 , 3 , 4 } ; int freq [ ] = { 10 , 5 , 20 , 100 } ; int i , n = arr . length ;
for ( i = 0 ; i < 5 ; i ++ ) System . out . println ( myRand ( arr , freq , n ) ) ; } }
import java . io . * ; class GFG {
static int calcAngle ( double h , double m ) {
if ( h < 0 m < 0 h > 12 m > 60 ) System . out . println ( " Wrong ▁ input " ) ; if ( h == 12 ) h = 0 ; if ( m == 60 ) { m = 0 ; h += 1 ; if ( h > 12 ) h = h - 12 ; }
int hour_angle = ( int ) ( 0.5 * ( h * 60 + m ) ) ; int minute_angle = ( int ) ( 6 * m ) ;
int angle = Math . abs ( hour_angle - minute_angle ) ;
angle = Math . min ( 360 - angle , angle ) ; return angle ; }
public static void main ( String [ ] args ) { System . out . println ( calcAngle ( 9 , 60 ) + " ▁ degree " ) ; System . out . println ( calcAngle ( 3 , 30 ) + " ▁ degree " ) ; } }
class GFG {
static int getSingle ( int arr [ ] , int n ) { int ones = 0 , twos = 0 ; int common_bit_mask ; for ( int i = 0 ; i < n ; i ++ ) {
twos = twos | ( ones & arr [ i ] ) ;
ones = ones ^ arr [ i ] ;
common_bit_mask = ~ ( ones & twos ) ;
ones &= common_bit_mask ;
twos &= common_bit_mask ; } return ones ; }
public static void main ( String args [ ] ) { int arr [ ] = { 3 , 3 , 2 , 3 } ; int n = arr . length ; System . out . println ( " The ▁ element ▁ with ▁ single ▁ occurrence ▁ is ▁ " + getSingle ( arr , n ) ) ; } }
class GFG { static final int INT_SIZE = 32 ; static int getSingle ( int arr [ ] , int n ) {
int result = 0 ; int x , sum ;
for ( int i = 0 ; i < INT_SIZE ; i ++ ) {
sum = 0 ; x = ( 1 << i ) ; for ( int j = 0 ; j < n ; j ++ ) { if ( ( arr [ j ] & x ) == 0 ) sum ++ ; }
if ( ( sum % 3 ) != 0 ) result |= x ; } return result ; }
public static void main ( String args [ ] ) { int arr [ ] = { 12 , 1 , 12 , 3 , 12 , 1 , 1 , 2 , 3 , 2 , 2 , 3 , 7 } ; int n = arr . length ; System . out . println ( " The ▁ element ▁ with ▁ single ▁ occurrence ▁ is ▁ " + getSingle ( arr , n ) ) ; } }
import java . io . * ; class GFG {
static int getLeftmostBit ( int n ) { int m = 0 ; while ( n > 1 ) { n = n >> 1 ; m ++ ; } return m ; }
static int getNextLeftmostBit ( int n , int m ) { int temp = 1 << m ; while ( n < temp ) { temp = temp >> 1 ; m -- ; } return m ; }
static int countSetBits ( int n ) {
int m = getLeftmostBit ( n ) ;
return countSetBits ( n , m ) ; } static int countSetBits ( int n , int m ) {
if ( n == 0 ) return 0 ;
m = getNextLeftmostBit ( n , m ) ;
if ( n == ( ( int ) 1 << ( m + 1 ) ) - 1 ) return ( int ) ( m + 1 ) * ( 1 << m ) ;
n = n - ( 1 << m ) ; return ( n + 1 ) + countSetBits ( n ) + m * ( 1 << ( m - 1 ) ) ; }
public static void main ( String [ ] args ) { int n = 17 ; System . out . println ( " Total ▁ set ▁ bit ▁ count ▁ is ▁ " + countSetBits ( n ) ) ; } }
class GFG { static int swapBits ( int x , int p1 , int p2 , int n ) {
int set1 = ( x >> p1 ) & ( ( 1 << n ) - 1 ) ;
int set2 = ( x >> p2 ) & ( ( 1 << n ) - 1 ) ;
int xor = ( set1 ^ set2 ) ;
xor = ( xor << p1 ) | ( xor << p2 ) ;
int result = x ^ xor ; return result ; }
public static void main ( String [ ] args ) { int res = swapBits ( 28 , 0 , 3 , 2 ) ; System . out . println ( " Result ▁ = ▁ " + res ) ; } }
class GFG { static int smallest ( int x , int y , int z ) { int c = 0 ; while ( x != 0 && y != 0 && z != 0 ) { x -- ; y -- ; z -- ; c ++ ; } return c ; }
public static void main ( String [ ] args ) { int x = 12 , y = 15 , z = 5 ; System . out . printf ( " Minimum ▁ of ▁ 3" + " ▁ numbers ▁ is ▁ % d " , smallest ( x , y , z ) ) ; } }
class GFG { static int CHAR_BIT = 8 ;
static int min ( int x , int y ) { return y + ( ( x - y ) & ( ( x - y ) >> ( ( Integer . SIZE / 8 ) * CHAR_BIT - 1 ) ) ) ; }
static int smallest ( int x , int y , int z ) { return Math . min ( x , Math . min ( y , z ) ) ; }
public static void main ( String [ ] args ) { int x = 12 , y = 15 , z = 5 ; System . out . println ( " Minimum ▁ of ▁ 3 ▁ numbers ▁ is ▁ " + smallest ( x , y , z ) ) ; } }
class GfG {
static int smallest ( int x , int y , int z ) {
if ( ( y / x ) != 1 ) return ( ( y / z ) != 1 ) ? y : z ; return ( ( x / z ) != 1 ) ? x : z ; }
public static void main ( String [ ] args ) { int x = 78 , y = 88 , z = 68 ; System . out . printf ( " Minimum ▁ of ▁ 3 ▁ numbers " + " ▁ is ▁ % d " , smallest ( x , y , z ) ) ; } }
import java . io . * ; class GFG { public static void changeToZero ( int a [ ] ) { a [ a [ 1 ] ] = a [ 1 - a [ 1 ] ] ; }
public static void main ( String args [ ] ) { int [ ] arr ; arr = new int [ 2 ] ; arr [ 0 ] = 1 ; arr [ 1 ] = 0 ; changeToZero ( arr ) ; System . out . println ( " arr [ 0 ] = ▁ " + arr [ 0 ] ) ; System . out . println ( " arr [ 1 ] = ▁ " + arr [ 1 ] ) ; } }
class GFG { static int addOne ( int x ) { int m = 1 ;
while ( ( int ) ( x & m ) >= 1 ) { x = x ^ m ; m <<= 1 ; }
x = x ^ m ; return x ; }
public static void main ( String [ ] args ) { System . out . println ( addOne ( 13 ) ) ; } }
class GFG { static int addOne ( int x ) { return ( - ( ~ x ) ) ; }
public static void main ( String [ ] args ) { System . out . printf ( " % d " , addOne ( 13 ) ) ; } }
class GFG {
static int fun ( int n ) { return n & ( n - 1 ) ; }
public static void main ( String arg [ ] ) { int n = 7 ; System . out . print ( " The ▁ number ▁ after ▁ unsetting ▁ " + " the ▁ rightmost ▁ set ▁ bit ▁ " + fun ( n ) ) ; } }
class GFG {
static int isPowerOfFour ( int n ) { if ( n == 0 ) return 0 ; while ( n != 1 ) { if ( n % 4 != 0 ) return 0 ; n = n / 4 ; } return 1 ; }
public static void main ( String [ ] args ) { int test_no = 64 ; if ( isPowerOfFour ( test_no ) == 1 ) System . out . println ( test_no + " ▁ is ▁ a ▁ power ▁ of ▁ 4" ) ; else System . out . println ( test_no + " is ▁ not ▁ a ▁ power ▁ of ▁ 4" ) ; } }
import java . io . * ; class GFG {
static int isPowerOfFour ( int n ) { int count = 0 ;
int x = n & ( n - 1 ) ; if ( n > 0 && x == 0 ) {
while ( n > 1 ) { n >>= 1 ; count += 1 ; }
return ( count % 2 == 0 ) ? 1 : 0 ; }
return 0 ; }
public static void main ( String [ ] args ) { int test_no = 64 ; if ( isPowerOfFour ( test_no ) > 0 ) System . out . println ( test_no + " ▁ is ▁ a ▁ power ▁ of ▁ 4" ) ; else System . out . println ( test_no + " ▁ is ▁ not ▁ a ▁ power ▁ of ▁ 4" ) ; } }
import java . io . * ; class GFG { static boolean isPowerOfFour ( int n ) { return n != 0 && ( ( n & ( n - 1 ) ) == 0 ) && ( n & 0xAAAAAAAA ) == 0 ; }
public static void main ( String [ ] args ) { int test_no = 64 ; if ( isPowerOfFour ( test_no ) ) System . out . println ( test_no + " ▁ is ▁ a ▁ power ▁ of ▁ 4" ) ; else System . out . println ( test_no + " ▁ is ▁ not ▁ a ▁ power ▁ of ▁ 4" ) ; } }
public class AWS {
static int min ( int x , int y ) { return y ^ ( ( x ^ y ) & - ( x << y ) ) ; }
static int max ( int x , int y ) { return x ^ ( ( x ^ y ) & - ( x << y ) ) ; }
public static void main ( String [ ] args ) { int x = 15 ; int y = 6 ; System . out . print ( " Minimum ▁ of ▁ " + x + " ▁ and ▁ " + y + " ▁ is ▁ " ) ; System . out . println ( min ( x , y ) ) ; System . out . print ( " Maximum ▁ of ▁ " + x + " ▁ and ▁ " + y + " ▁ is ▁ " ) ; System . out . println ( max ( x , y ) ) ; } }
class GFG { static int CHAR_BIT = 4 ; static int INT_BIT = 8 ;
static int min ( int x , int y ) { return y + ( ( x - y ) & ( ( x - y ) >> ( INT_BIT * CHAR_BIT - 1 ) ) ) ; }
static int max ( int x , int y ) { return x - ( ( x - y ) & ( ( x - y ) >> ( INT_BIT * CHAR_BIT - 1 ) ) ) ; }
public static void main ( String [ ] args ) { int x = 15 ; int y = 6 ; System . out . println ( " Minimum ▁ of ▁ " + x + " ▁ and ▁ " + y + " ▁ is ▁ " + min ( x , y ) ) ; System . out . println ( " Maximum ▁ of ▁ " + x + " ▁ and ▁ " + y + " ▁ is ▁ " + max ( x , y ) ) ; } }
class GFG { public static int getFirstSetBitPos ( int n ) { return ( int ) ( ( Math . log10 ( n & - n ) ) / Math . log10 ( 2 ) ) + 1 ; }
public static void main ( String [ ] args ) { int n = 12 ; System . out . println ( getFirstSetBitPos ( n ) ) ; } }
public class GFG {
static void bin ( long n ) { long i ; System . out . print ( "0" ) ; for ( i = 1 << 30 ; i > 0 ; i = i / 2 ) { if ( ( n & i ) != 0 ) { System . out . print ( "1" ) ; } else { System . out . print ( "0" ) ; } } }
public static void main ( String [ ] args ) { bin ( 7 ) ; System . out . println ( ) ; bin ( 4 ) ; } }
class GFG {
static int swapBits ( int x ) {
int even_bits = x & 0xAAAAAAAA ;
int odd_bits = x & 0x55555555 ;
even_bits >>= 1 ;
odd_bits <<= 1 ;
return ( even_bits odd_bits ) ; }
public static void main ( String [ ] args ) {
int x = 23 ;
System . out . println ( swapBits ( x ) ) ; } }
class GFG {
static boolean isPowerOfTwo ( int n ) { return ( n > 0 && ( ( n & ( n - 1 ) ) == 0 ) ) ? true : false ; }
static int findPosition ( int n ) { if ( ! isPowerOfTwo ( n ) ) return - 1 ; int i = 1 , pos = 1 ;
while ( ( i & n ) == 0 ) {
i = i << 1 ;
++ pos ; } return pos ; }
public static void main ( String [ ] args ) { int n = 16 ; int pos = findPosition ( n ) ; if ( pos == - 1 ) System . out . println ( " n ▁ = ▁ " + n + " , ▁ Invalid ▁ number " ) ; else System . out . println ( " n ▁ = ▁ " + n + " , ▁ Position ▁ " + pos ) ; n = 12 ; pos = findPosition ( n ) ; if ( pos == - 1 ) System . out . println ( " n ▁ = ▁ " + n + " , ▁ Invalid ▁ number " ) ; else System . out . println ( " n ▁ = ▁ " + n + " , ▁ Position ▁ " + pos ) ; n = 128 ; pos = findPosition ( n ) ; if ( pos == - 1 ) System . out . println ( " n ▁ = ▁ " + n + " , ▁ Invalid ▁ number " ) ; else System . out . println ( " n ▁ = ▁ " + n + " , ▁ Position ▁ " + pos ) ; } }
class GFG {
static boolean isPowerOfTwo ( int n ) { return n > 0 && ( ( n & ( n - 1 ) ) == 0 ) ; }
static int findPosition ( int n ) { if ( ! isPowerOfTwo ( n ) ) return - 1 ; int count = 0 ;
while ( n > 0 ) { n = n >> 1 ;
++ count ; } return count ; }
public static void main ( String [ ] args ) { int n = 0 ; int pos = findPosition ( n ) ; if ( pos == - 1 ) System . out . println ( " n ▁ = ▁ " + n + " , ▁ Invalid ▁ number " ) ; else System . out . println ( " n ▁ = ▁ " + n + " , ▁ Position ▁ " + pos ) ; n = 12 ; pos = findPosition ( n ) ; if ( pos == - 1 ) System . out . println ( " n ▁ = ▁ " + n + " , ▁ Invalid ▁ number " ) ; else System . out . println ( " n ▁ = ▁ " + n + " , ▁ Position ▁ " + pos ) ; n = 128 ; pos = findPosition ( n ) ; if ( pos == - 1 ) System . out . println ( " n ▁ = ▁ " + n + " , ▁ Invalid ▁ number " ) ; else System . out . println ( " n ▁ = ▁ " + n + " , ▁ Position ▁ " + pos ) ; } }
import java . io . * ; class GFG { public static void main ( String [ ] args ) { int x = 10 ; int y = 5 ;
x = x * y ;
y = x / y ;
x = x / y ; System . out . println ( " After ▁ swaping : " + " ▁ x ▁ = ▁ " + x + " , ▁ y ▁ = ▁ " + y ) ; } }
import java . io . * ; public class GFG { public static void main ( String a [ ] ) { int x = 10 ; int y = 5 ;
x = x ^ y ;
y = x ^ y ;
x = x ^ y ; System . out . println ( " After ▁ swap : ▁ x ▁ = ▁ " + x + " , ▁ y ▁ = ▁ " + y ) ; } }
class GFG {
static void swap ( int [ ] xp , int [ ] yp ) { xp [ 0 ] = xp [ 0 ] ^ yp [ 0 ] ; yp [ 0 ] = xp [ 0 ] ^ yp [ 0 ] ; xp [ 0 ] = xp [ 0 ] ^ yp [ 0 ] ; }
public static void main ( String [ ] args ) { int [ ] x = { 10 } ; swap ( x , x ) ; System . out . println ( " After ▁ swap ( & x , ▁ & x ) : ▁ x ▁ = ▁ " + x [ 0 ] ) ; } }
import java . io . * ; class NextGreatest {
static void nextGreatest ( int arr [ ] ) { int size = arr . length ;
int max_from_right = arr [ size - 1 ] ;
arr [ size - 1 ] = - 1 ;
for ( int i = size - 2 ; i >= 0 ; i -- ) {
int temp = arr [ i ] ;
arr [ i ] = max_from_right ;
if ( max_from_right < temp ) max_from_right = temp ; } }
static void printArray ( int arr [ ] ) { for ( int i = 0 ; i < arr . length ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 16 , 17 , 4 , 3 , 5 , 2 } ; nextGreatest ( arr ) ; System . out . println ( " The ▁ modified ▁ array : " ) ; printArray ( arr ) ; } }
class MaximumDiffrence {
int maxDiff ( int arr [ ] , int arr_size ) { int max_diff = arr [ 1 ] - arr [ 0 ] ; int i , j ; for ( i = 0 ; i < arr_size ; i ++ ) { for ( j = i + 1 ; j < arr_size ; j ++ ) { if ( arr [ j ] - arr [ i ] > max_diff ) max_diff = arr [ j ] - arr [ i ] ; } } return max_diff ; }
public static void main ( String [ ] args ) { MaximumDifference maxdif = new MaximumDifference ( ) ; int arr [ ] = { 1 , 2 , 90 , 10 , 110 } ;
System . out . println ( " Maximum ▁ difference ▁ is ▁ " + maxdif . maxDiff ( arr , 5 ) ) ; } }
class Main {
static int findMaximum ( int arr [ ] , int low , int high ) { int max = arr [ low ] ; int i ; for ( i = low ; i <= high ; i ++ ) { if ( arr [ i ] > max ) max = arr [ i ] ; } return max ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 30 , 40 , 50 , 60 , 70 , 23 , 20 } ; int n = arr . length ; System . out . println ( " The ▁ maximum ▁ element ▁ is ▁ " + findMaximum ( arr , 0 , n - 1 ) ) ; } }
class Main { static int findMaximum ( int arr [ ] , int low , int high ) {
if ( low == high ) return arr [ low ] ;
if ( ( high == low + 1 ) && arr [ low ] >= arr [ high ] ) return arr [ low ] ;
if ( ( high == low + 1 ) && arr [ low ] < arr [ high ] ) return arr [ high ] ; int mid = ( low + high ) / 2 ;
if ( arr [ mid ] > arr [ mid + 1 ] && arr [ mid ] > arr [ mid - 1 ] ) return arr [ mid ] ;
if ( arr [ mid ] > arr [ mid + 1 ] && arr [ mid ] < arr [ mid - 1 ] ) return findMaximum ( arr , low , mid - 1 ) ;
else return findMaximum ( arr , mid + 1 , high ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 3 , 50 , 10 , 9 , 7 , 6 } ; int n = arr . length ; System . out . println ( " The ▁ maximum ▁ element ▁ is ▁ " + findMaximum ( arr , 0 , n - 1 ) ) ; } }
class CountSmaller { void constructLowerArray ( int arr [ ] , int countSmaller [ ] , int n ) { int i , j ;
for ( i = 0 ; i < n ; i ++ ) countSmaller [ i ] = 0 ; for ( i = 0 ; i < n ; i ++ ) { for ( j = i + 1 ; j < n ; j ++ ) { if ( arr [ j ] < arr [ i ] ) countSmaller [ i ] ++ ; } } }
void printArray ( int arr [ ] , int size ) { int i ; for ( i = 0 ; i < size ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; System . out . println ( " " ) ; }
public static void main ( String [ ] args ) { CountSmaller small = new CountSmaller ( ) ; int arr [ ] = { 12 , 10 , 5 , 4 , 2 , 20 , 6 , 1 , 0 , 2 } ; int n = arr . length ; int low [ ] = new int [ n ] ; small . constructLowerArray ( arr , low , n ) ; small . printArray ( low , n ) ; } }
import java . util . * ; class Main {
static int segregate ( int arr [ ] , int size ) { int j = 0 , i ; for ( i = 0 ; i < size ; i ++ ) { if ( arr [ i ] <= 0 ) { int temp ; temp = arr [ i ] ; arr [ i ] = arr [ j ] ; arr [ j ] = temp ;
j ++ ; } } return j ; }
static int findMissingPositive ( int arr [ ] , int size ) { int i ;
for ( i = 0 ; i < size ; i ++ ) { int x = Math . abs ( arr [ i ] ) ; if ( x - 1 < size && arr [ x - 1 ] > 0 ) arr [ x - 1 ] = - arr [ x - 1 ] ; }
for ( i = 0 ; i < size ; i ++ ) if ( arr [ i ] > 0 )
return i + 1 ; return size + 1 ; }
static int findMissing ( int arr [ ] , int size ) {
int shift = segregate ( arr , size ) ; int arr2 [ ] = new int [ size - shift ] ; int j = 0 ; for ( int i = shift ; i < size ; i ++ ) { arr2 [ j ] = arr [ i ] ; j ++ ; }
return findMissingPositive ( arr2 , j ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 0 , 10 , 2 , - 10 , - 20 } ; int arr_size = arr . length ; int missing = findMissing ( arr , arr_size ) ; System . out . println ( " The ▁ smallest ▁ positive ▁ missing ▁ number ▁ is ▁ " + missing ) ; } }
import java . util . * ; import java . util . Arrays ; class GFG { public static List < Integer >
findDisappearedNumbers ( int [ ] nums ) { for ( int i = 0 ; i < nums . length ; i ++ ) { int index = Math . abs ( nums [ i ] ) ; if ( nums [ index - 1 ] > 0 ) { nums [ index - 1 ] *= - 1 ; } } List < Integer > res = new ArrayList < > ( ) ; for ( int i = 0 ; i < nums . length ; i ++ ) { if ( nums [ i ] > 0 ) { res . add ( i + 1 ) ; } } return res ; }
public static void main ( String [ ] args ) { int [ ] a = { 1 , 2 , 4 , 5 , 6 } ; System . out . println ( findDisappearedNumbers ( a ) ) ; } }
import java . io . * ; class GFG { static void printTwoElements ( int arr [ ] , int size ) { int i ; System . out . print ( " The ▁ repeating ▁ element ▁ is ▁ " ) ; for ( i = 0 ; i < size ; i ++ ) { int abs_val = Math . abs ( arr [ i ] ) ; if ( arr [ abs_val - 1 ] > 0 ) arr [ abs_val - 1 ] = - arr [ abs_val - 1 ] ; else System . out . println ( abs_val ) ; } System . out . print ( " And ▁ the ▁ missing ▁ element ▁ is ▁ " ) ; for ( i = 0 ; i < size ; i ++ ) { if ( arr [ i ] > 0 ) System . out . println ( i + 1 ) ; } }
public static void main ( String [ ] args ) { int arr [ ] = { 7 , 3 , 4 , 5 , 5 , 6 , 2 } ; int n = arr . length ; printTwoElements ( arr , n ) ; } }
import java . io . * ; class GFG { static int x , y ;
static void getTwoElements ( int arr [ ] , int n ) {
int xor1 ;
int set_bit_no ; int i ; x = 0 ; y = 0 ; xor1 = arr [ 0 ] ;
for ( i = 1 ; i < n ; i ++ ) xor1 = xor1 ^ arr [ i ] ;
for ( i = 1 ; i <= n ; i ++ ) xor1 = xor1 ^ i ;
set_bit_no = xor1 & ~ ( xor1 - 1 ) ;
for ( i = 0 ; i < n ; i ++ ) { if ( ( arr [ i ] & set_bit_no ) != 0 )
x = x ^ arr [ i ] ; else
y = y ^ arr [ i ] ; } for ( i = 1 ; i <= n ; i ++ ) { if ( ( i & set_bit_no ) != 0 )
x = x ^ i ; else
y = y ^ i ; }
}
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 3 , 4 , 5 , 1 , 6 , 2 } ; int n = arr . length ; getTwoElements ( arr , n ) ; System . out . println ( " ▁ The ▁ missing ▁ element ▁ is ▁ " + x + " and ▁ the ▁ " + " repeating ▁ number ▁ is ▁ " + y ) ; } }
class FindFourElements {
void findFourElements ( int A [ ] , int n , int X ) {
for ( int i = 0 ; i < n - 3 ; i ++ ) {
for ( int j = i + 1 ; j < n - 2 ; j ++ ) {
for ( int k = j + 1 ; k < n - 1 ; k ++ ) {
for ( int l = k + 1 ; l < n ; l ++ ) { if ( A [ i ] + A [ j ] + A [ k ] + A [ l ] == X ) System . out . print ( A [ i ] + " ▁ " + A [ j ] + " ▁ " + A [ k ] + " ▁ " + A [ l ] ) ; } } } } }
public static void main ( String [ ] args ) { FindFourElements findfour = new FindFourElements ( ) ; int A [ ] = { 10 , 20 , 30 , 40 , 1 , 2 } ; int n = A . length ; int X = 91 ; findfour . findFourElements ( A , n , X ) ; } }
class GFG {
static int minDistance ( int arr [ ] , int n ) { int maximum_element = arr [ 0 ] ; int min_dis = n ; int index = 0 ; for ( int i = 1 ; i < n ; i ++ ) {
if ( maximum_element == arr [ i ] ) { min_dis = Math . min ( min_dis , ( i - index ) ) ; index = i ; }
else if ( maximum_element < arr [ i ] ) { maximum_element = arr [ i ] ; min_dis = n ; index = i ; }
else continue ; } return min_dis ; }
public static void main ( String [ ] args ) { int arr [ ] = { 6 , 3 , 1 , 3 , 6 , 4 , 6 } ; int n = arr . length ; System . out . print ( " Minimum ▁ distance ▁ = ▁ " + minDistance ( arr , n ) ) ; } }
class Node { int data ; Node next ; Node ( int d ) { data = d ; next = null ; } }
public void push ( int new_data ) {
Node new_node = new Node ( new_data ) ;
new_node . next = head ;
head = new_node ; }
public boolean search ( Node head , int x ) {
if ( head == null ) return false ;
if ( head . data == x ) return true ;
return search ( head . next , x ) ; }
public static void main ( String args [ ] ) {
LinkedList llist = new LinkedList ( ) ;
llist . push ( 10 ) ; llist . push ( 30 ) ; llist . push ( 11 ) ; llist . push ( 21 ) ; llist . push ( 14 ) ; if ( llist . search ( llist . head , 21 ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
static Node deleteAlt ( Node head ) { if ( head == null ) return ; Node node = head . next ; if ( node == null ) return ;
head . next = node . next ;
head . next = deleteAlt ( head . next ) ; }
static void AlternatingSplit ( Node source , Node aRef , Node bRef ) { Node aDummy = new Node ( ) ; Node aTail = aDummy ;
Node bDummy = new Node ( ) ; Node bTail = bDummy ;
Node current = source ; aDummy . next = null ; bDummy . next = null ; while ( current != null ) { MoveNode ( ( aTail . next ) , current ) ;
aTail = aTail . next ;
if ( current != null ) { MoveNode ( ( bTail . next ) , current ) ; bTail = bTail . next ; } } aRef = aDummy . next ; bRef = bDummy . next ; }
class Node { int data ; Node next ; Node ( int d ) { data = d ; next = null ; } }
boolean areIdentical ( LinkedList listb ) { Node a = this . head , b = listb . head ; while ( a != null && b != null ) { if ( a . data != b . data ) return false ;
a = a . next ; b = b . next ; }
return ( a == null && b == null ) ; }
void push ( int new_data ) {
Node new_node = new Node ( new_data ) ;
new_node . next = head ;
head = new_node ; }
public static void main ( String args [ ] ) { LinkedList llist1 = new LinkedList ( ) ; LinkedList llist2 = new LinkedList ( ) ;
llist1 . push ( 1 ) ; llist1 . push ( 2 ) ; llist1 . push ( 3 ) ; llist2 . push ( 1 ) ; llist2 . push ( 2 ) ; llist2 . push ( 3 ) ; if ( llist1 . areIdentical ( llist2 ) == true ) System . out . println ( " Identical ▁ " ) ; else System . out . println ( " Not ▁ identical ▁ " ) ; } }
boolean areIdenticalRecur ( Node a , Node b ) {
if ( a == null && b == null ) return true ;
if ( a != null && b != null ) return ( a . data == b . data ) && areIdenticalRecur ( a . next , b . next ) ;
return false ; }
class LinkedList {
class Node { int data ; Node next ; Node ( int d ) { data = d ; next = null ; } } void sortList ( ) {
int count [ ] = { 0 , 0 , 0 } ; Node ptr = head ;
while ( ptr != null ) { count [ ptr . data ] ++ ; ptr = ptr . next ; } int i = 0 ; ptr = head ;
while ( ptr != null ) { if ( count [ i ] == 0 ) i ++ ; else { ptr . data = i ; -- count [ i ] ; ptr = ptr . next ; } } }
public void push ( int new_data ) {
Node new_node = new Node ( new_data ) ;
new_node . next = head ;
head = new_node ; }
void printList ( ) { Node temp = head ; while ( temp != null ) { System . out . print ( temp . data + " ▁ " ) ; temp = temp . next ; } System . out . println ( ) ; }
public static void main ( String args [ ] ) { LinkedList llist = new LinkedList ( ) ;
llist . push ( 0 ) ; llist . push ( 1 ) ; llist . push ( 0 ) ; llist . push ( 2 ) ; llist . push ( 1 ) ; llist . push ( 1 ) ; llist . push ( 2 ) ; llist . push ( 1 ) ; llist . push ( 2 ) ; System . out . println ( " Linked ▁ List ▁ before ▁ sorting " ) ; llist . printList ( ) ; llist . sortList ( ) ; System . out . println ( " Linked ▁ List ▁ after ▁ sorting " ) ; llist . printList ( ) ; } }
static class List { public int data ; public List next ; public List child ; } ;
class GfG {
static class Node { int data ; Node next ; }
static Node newNode ( int key ) { Node temp = new Node ( ) ; temp . data = key ; temp . next = null ; return temp ; }
static Node rearrangeEvenOdd ( Node head ) {
if ( head == null ) return null ;
Node odd = head ; Node even = head . next ;
Node evenFirst = even ; while ( 1 == 1 ) {
if ( odd == null || even == null || ( even . next ) == null ) { odd . next = evenFirst ; break ; }
odd . next = even . next ; odd = even . next ;
if ( odd . next == null ) { even . next = null ; odd . next = evenFirst ; break ; }
even . next = odd . next ; even = odd . next ; } return head ; }
static void printlist ( Node node ) { while ( node != null ) { System . out . print ( node . data + " - > " ) ; node = node . next ; } System . out . println ( " NULL " ) ; }
public static void main ( String [ ] args ) { Node head = newNode ( 1 ) ; head . next = newNode ( 2 ) ; head . next . next = newNode ( 3 ) ; head . next . next . next = newNode ( 4 ) ; head . next . next . next . next = newNode ( 5 ) ; System . out . println ( " Given ▁ Linked ▁ List " ) ; printlist ( head ) ; head = rearrangeEvenOdd ( head ) ; System . out . println ( " Modified ▁ Linked ▁ List " ) ; printlist ( head ) ; } }
class GFG {
static class Node { int data ; Node next ; } ;
static void deleteLast ( Node head , int x ) { Node temp = head , ptr = null ; while ( temp != null ) {
if ( temp . data == x ) ptr = temp ; temp = temp . next ; }
if ( ptr != null && ptr . next == null ) { temp = head ; while ( temp . next != ptr ) temp = temp . next ; temp . next = null ; }
if ( ptr != null && ptr . next != null ) { ptr . data = ptr . next . data ; temp = ptr . next ; ptr . next = ptr . next . next ; System . gc ( ) ; } }
static Node newNode ( int x ) { Node node = new Node ( ) ; node . data = x ; node . next = null ; return node ; }
static void display ( Node head ) { Node temp = head ; if ( head == null ) { System . out . print ( "nullNEW_LINE"); return ; } while ( temp != null ) { System . out . printf ( " % d ▁ - - > ▁ " , temp . data ) ; temp = temp . next ; } System . out . print ( "nullNEW_LINE"); }
public static void main ( String [ ] args ) { Node head = newNode ( 1 ) ; head . next = newNode ( 2 ) ; head . next . next = newNode ( 3 ) ; head . next . next . next = newNode ( 4 ) ; head . next . next . next . next = newNode ( 5 ) ; head . next . next . next . next . next = newNode ( 4 ) ; head . next . next . next . next . next . next = newNode ( 4 ) ; System . out . print ( " Created ▁ Linked ▁ list : ▁ " ) ; display ( head ) ; deleteLast ( head , 4 ) ; System . out . print ( " List ▁ after ▁ deletion ▁ of ▁ 4 : ▁ " ) ; display ( head ) ; } }
import java . io . * ; class GfG {
static class Node { int data ; Node next ; }
static int LinkedListLength ( Node head ) { while ( head != null && head . next != null ) { head = head . next . next ; } if ( head == null ) return 0 ; return 1 ; }
static void push ( Node head , int info ) {
Node node = new Node ( ) ;
node . data = info ;
node . next = ( head ) ;
( head ) = node ; }
public static void main ( String [ ] args ) { Node head = null ;
push ( head , 4 ) ; push ( head , 5 ) ; push ( head , 7 ) ; push ( head , 2 ) ; push ( head , 9 ) ; push ( head , 6 ) ; push ( head , 1 ) ; push ( head , 2 ) ; push ( head , 0 ) ; push ( head , 5 ) ; push ( head , 5 ) ; int check = LinkedListLength ( head ) ;
if ( check == 0 ) { System . out . println ( " Odd " ) ; } else { System . out . println ( " Even " ) ; } } }
Node SortedMerge ( Node a , Node b ) { Node result = null ;
Node lastPtrRef = result ; while ( 1 ) { if ( a == null ) { lastPtrRef = b ; break ; } else if ( b == null ) { lastPtrRef = a ; break ; } if ( a . data <= b . data ) { MoveNode ( lastPtrRef , a ) ; } else { MoveNode ( lastPtrRef , b ) ; }
lastPtrRef = ( ( lastPtrRef ) . next ) ; } return ( result ) ; }
public class GFG {
static class Node { int data ; Node next ; Node ( int data ) { this . data = data ; next = null ; } } static Node head ;
static void setMiddleHead ( ) { if ( head == null ) return ;
Node one_node = head ;
Node two_node = head ;
Node prev = null ; while ( two_node != null && two_node . next != null ) {
prev = one_node ;
two_node = two_node . next . next ;
one_node = one_node . next ; }
prev . next = prev . next . next ; one_node . next = head ; head = one_node ; }
static void push ( int new_data ) {
Node new_node = new Node ( new_data ) ;
new_node . next = head ;
head = new_node ; }
static void printList ( Node ptr ) { while ( ptr != null ) { System . out . print ( ptr . data + " ▁ " ) ; ptr = ptr . next ; } System . out . println ( ) ; }
public static void main ( String args [ ] ) {
head = null ; int i ; for ( i = 5 ; i > 0 ; i -- ) push ( i ) ; System . out . print ( " ▁ list ▁ before : ▁ " ) ; printList ( head ) ; setMiddleHead ( ) ; System . out . print ( " ▁ list ▁ After : ▁ " ) ; printList ( head ) ; } }
public void InsertAfter ( Node prev_Node , int new_data ) {
if ( prev_Node == null ) { System . out . println ( " The ▁ given ▁ previous ▁ node ▁ cannot ▁ be ▁ NULL ▁ " ) ; return ; }
Node new_node = new Node ( new_data ) ;
new_node . next = prev_Node . next ;
prev_Node . next = new_node ;
new_node . prev = prev_Node ;
if ( new_node . next != null ) new_node . next . prev = new_node ; }
class Node { int data ; Node left , right ; Node ( int item ) { data = item ; left = right = null ; } } class BinaryTree { Node root ; void printKDistant ( Node node , int k ) { if ( node == null k < 0 ) return ; if ( k == 0 ) { System . out . print ( node . data + " ▁ " ) ; return ; } printKDistant ( node . left , k - 1 ) ; printKDistant ( node . right , k - 1 ) ; }
public static void main ( String args [ ] ) { BinaryTree tree = new BinaryTree ( ) ;
tree . root = new Node ( 1 ) ; tree . root . left = new Node ( 2 ) ; tree . root . right = new Node ( 3 ) ; tree . root . left . left = new Node ( 4 ) ; tree . root . left . right = new Node ( 5 ) ; tree . root . right . left = new Node ( 8 ) ; tree . printKDistant ( tree . root , 2 ) ; } }
class GFG { static final int COUNT = 10 ;
static class Node { int data ; Node left , right ;
Node ( int data ) { this . data = data ; this . left = null ; this . right = null ; } } ;
static void print2DUtil ( Node root , int space ) {
if ( root == null ) return ;
space += COUNT ;
print2DUtil ( root . right , space ) ;
System . out . print ( "NEW_LINE"); for ( int i = COUNT ; i < space ; i ++ ) System . out . print ( " ▁ " ) ; System . out . print ( root . data + "NEW_LINE");
print2DUtil ( root . left , space ) ; }
static void print2D ( Node root ) {
print2DUtil ( root , 0 ) ; }
public static void main ( String args [ ] ) { Node root = new Node ( 1 ) ; root . left = new Node ( 2 ) ; root . right = new Node ( 3 ) ; root . left . left = new Node ( 4 ) ; root . left . right = new Node ( 5 ) ; root . right . left = new Node ( 6 ) ; root . right . right = new Node ( 7 ) ; root . left . left . left = new Node ( 8 ) ; root . left . left . right = new Node ( 9 ) ; root . left . right . left = new Node ( 10 ) ; root . left . right . right = new Node ( 11 ) ; root . right . left . left = new Node ( 12 ) ; root . right . left . right = new Node ( 13 ) ; root . right . right . left = new Node ( 14 ) ; root . right . right . right = new Node ( 15 ) ; print2D ( root ) ; } }
class Node { int data ; Node left , right ; public Node ( int item ) { data = item ; left = right = null ; } }
void leftViewUtil ( Node node , int level ) {
if ( node == null ) return ;
if ( max_level < level ) { System . out . print ( " ▁ " + node . data ) ; max_level = level ; }
leftViewUtil ( node . left , level + 1 ) ; leftViewUtil ( node . right , level + 1 ) ; }
void leftView ( ) { leftViewUtil ( root , 1 ) ; }
public static void main ( String args [ ] ) { BinaryTree tree = new BinaryTree ( ) ; tree . root = new Node ( 12 ) ; tree . root . left = new Node ( 10 ) ; tree . root . right = new Node ( 30 ) ; tree . root . right . left = new Node ( 25 ) ; tree . root . right . right = new Node ( 40 ) ; tree . leftView ( ) ; } }
class GFG {
public static int cntRotations ( char s [ ] , int n ) { int lh = 0 , rh = 0 , i , ans = 0 ;
for ( i = 0 ; i < n / 2 ; ++ i ) if ( s [ i ] == ' a ' s [ i ] == ' e ' s [ i ] == ' i ' s [ i ] == ' o ' s [ i ] == ' u ' ) { lh ++ ; }
for ( i = n / 2 ; i < n ; ++ i ) if ( s [ i ] == ' a ' s [ i ] == ' e ' s [ i ] == ' i ' s [ i ] == ' o ' s [ i ] == ' u ' ) { rh ++ ; }
if ( lh > rh ) ans ++ ;
for ( i = 1 ; i < n ; ++ i ) { if ( s [ i - 1 ] == ' a ' s [ i - 1 ] == ' e ' s [ i - 1 ] == ' i ' s [ i - 1 ] == ' o ' s [ i - 1 ] == ' u ' ) { rh ++ ; lh -- ; } if ( s [ ( i - 1 + n / 2 ) % n ] == ' a ' || s [ ( i - 1 + n / 2 ) % n ] == ' e ' || s [ ( i - 1 + n / 2 ) % n ] == ' i ' || s [ ( i - 1 + n / 2 ) % n ] == ' o ' || s [ ( i - 1 + n / 2 ) % n ] == ' u ' ) { rh -- ; lh ++ ; } if ( lh > rh ) ans ++ ; }
return ans ; }
public static void main ( String [ ] args ) { char s [ ] = { ' a ' , ' b ' , ' e ' , ' c ' , ' i ' , ' d ' , ' f ' , ' t ' } ; int n = s . length ;
System . out . println ( cntRotations ( s , n ) ) ; } }
import java . util . * ; class GFG {
static class Node { int data ; Node next ; } ; static Node tail ;
static Node rotateHelper ( Node blockHead , Node blockTail , int d , int k ) { if ( d == 0 ) return blockHead ;
if ( d > 0 ) { Node temp = blockHead ; for ( int i = 1 ; temp . next . next != null && i < k - 1 ; i ++ ) temp = temp . next ; blockTail . next = blockHead ; tail = temp ; return rotateHelper ( blockTail , temp , d - 1 , k ) ; }
if ( d < 0 ) { blockTail . next = blockHead ; tail = blockHead ; return rotateHelper ( blockHead . next , blockHead , d + 1 , k ) ; } return blockHead ; }
static Node rotateByBlocks ( Node head , int k , int d ) {
if ( head == null head . next == null ) return head ;
if ( d == 0 ) return head ; Node temp = head ; tail = null ;
int i ; for ( i = 1 ; temp . next != null && i < k ; i ++ ) temp = temp . next ;
Node nextBlock = temp . next ;
if ( i < k ) head = rotateHelper ( head , temp , d % k , i ) ; else head = rotateHelper ( head , temp , d % k , k ) ;
tail . next = rotateByBlocks ( nextBlock , k , d % k ) ;
return head ; }
static Node push ( Node head_ref , int new_data ) { Node new_node = new Node ( ) ; new_node . data = new_data ; new_node . next = head_ref ; head_ref = new_node ; return head_ref ; }
static void printList ( Node node ) { while ( node != null ) { System . out . print ( node . data + " ▁ " ) ; node = node . next ; } }
public static void main ( String [ ] args ) {
Node head = null ;
for ( int i = 9 ; i > 0 ; i -= 1 ) head = push ( head , i ) ; System . out . print ( "Given linked list NEW_LINE"); printList ( head ) ;
int k = 3 , d = 2 ; head = rotateByBlocks ( head , k , d ) ; System . out . print ( " Rotated by blocks Linked list "); printList ( head ) ; } }
static void DeleteFirst ( Node head ) { Node previous = head , firstNode = head ;
if ( head == null ) { System . out . printf ( " List is empty "); return ; }
if ( previous . next == previous ) { head = null ; return ; }
while ( previous . next != head ) { previous = previous . next ; }
previous . next = firstNode . next ;
head = previous . next ; System . gc ( ) ; return ; }
static Node DeleteLast ( Node head ) { Node current = head , temp = head , previous = null ;
if ( head == null ) { System . out . printf ( " List is empty "); return null ; }
if ( current . next == current ) { head = null ; return null ; }
while ( current . next != head ) { previous = current ; current = current . next ; } previous . next = current . next ; head = previous . next ; return head ; }
import java . util . * ; class GFG {
static void countSubarrays ( int arr [ ] , int n ) {
int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { int sum = 0 ; for ( int j = i ; j < n ; j ++ ) {
if ( ( j - i ) % 2 == 0 ) sum += arr [ j ] ;
else sum -= arr [ j ] ;
if ( sum == 0 ) count ++ ; } }
System . out . println ( count ) ; }
public static void main ( String [ ] args ) {
int arr [ ] = { 2 , 4 , 6 , 4 , 2 } ;
int n = arr . length ;
countSubarrays ( arr , n ) ; } }
static void print ( int n ) { if ( n < 0 ) return ; System . out . print ( " ▁ " + n ) ;
print ( n - 1 ) ; }
import java . io . * ; class GFG {
static void printAlter ( int [ ] arr , int N ) {
for ( int currIndex = 0 ; currIndex < N ; currIndex ++ ) {
if ( currIndex % 2 == 0 ) { System . out . print ( arr [ currIndex ] + " ▁ " ) ; } } }
public static void main ( String [ ] args ) { int [ ] arr = { 1 , 2 , 3 , 4 , 5 } ; int N = arr . length ; printAlter ( arr , N ) ; } }
class GFG {
static void reverse ( int arr [ ] , int start , int end ) {
int mid = ( end - start + 1 ) / 2 ;
for ( int i = 0 ; i < mid ; i ++ ) {
int temp = arr [ start + i ] ;
arr [ start + i ] = arr [ end - i ] ;
arr [ end - i ] = temp ; } return ; }
static void shuffleArrayUtil ( int arr [ ] , int start , int end ) { int i ;
int l = end - start + 1 ;
if ( l == 2 ) return ;
int mid = start + l / 2 ;
if ( l % 4 > 0 ) {
mid -= 1 ; }
int mid1 = start + ( mid - start ) / 2 ; int mid2 = mid + ( end + 1 - mid ) / 2 ;
reverse ( arr , mid1 , mid2 - 1 ) ;
reverse ( arr , mid1 , mid - 1 ) ;
reverse ( arr , mid , mid2 - 1 ) ;
shuffleArrayUtil ( arr , start , mid - 1 ) ; shuffleArrayUtil ( arr , mid , end ) ; }
static void shuffleArray ( int arr [ ] , int N , int start , int end ) {
shuffleArrayUtil ( arr , start , end ) ;
for ( int i = 0 ; i < N ; i ++ ) System . out . printf ( " % d ▁ " , arr [ i ] ) ; }
public static void main ( String [ ] args ) {
int arr [ ] = { 1 , 3 , 5 , 2 , 4 , 6 } ;
int N = arr . length ;
shuffleArray ( arr , N , 0 , N - 1 ) ; } }
import java . util . * ; class GFG {
public static boolean canMadeEqual ( int [ ] A , int [ ] B , int n ) {
Arrays . sort ( A ) ; Arrays . sort ( B ) ;
for ( int i = 0 ; i < n ; i ++ ) { if ( A [ i ] != B [ i ] ) { return false ; } } return true ; }
public static void main ( String [ ] args ) { int A [ ] = { 1 , 2 , 3 } ; int B [ ] = { 1 , 3 , 2 } ; int n = A . length ; if ( canMadeEqual ( A , B , n ) ) System . out . print ( " Yes " ) ; else System . out . print ( " No " ) ; } }
public class GFG {
static void merge ( int arr [ ] , int start , int mid , int end ) { int start2 = mid + 1 ;
if ( arr [ mid ] <= arr [ start2 ] ) { return ; }
while ( start <= mid && start2 <= end ) {
if ( arr [ start ] <= arr [ start2 ] ) { start ++ ; } else { int value = arr [ start2 ] ; int index = start2 ;
while ( index != start ) { arr [ index ] = arr [ index - 1 ] ; index -- ; } arr [ start ] = value ;
start ++ ; mid ++ ; start2 ++ ; } } }
static void mergeSort ( int arr [ ] , int l , int r ) { if ( l < r ) {
int m = l + ( r - l ) / 2 ;
mergeSort ( arr , l , m ) ; mergeSort ( arr , m + 1 , r ) ; merge ( arr , l , m , r ) ; } }
static void printArray ( int A [ ] , int size ) { int i ; for ( i = 0 ; i < size ; i ++ ) System . out . print ( A [ i ] + " ▁ " ) ; System . out . println ( ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 12 , 11 , 13 , 5 , 6 , 7 } ; int arr_size = arr . length ; mergeSort ( arr , 0 , arr_size - 1 ) ; printArray ( arr , arr_size ) ; } }
class GFG { static void swap ( int [ ] arr , int i , int j ) { int temp = arr [ i ] ; arr [ i ] = arr [ j ] ; arr [ j ] = temp ; } static void dualPivotQuickSort ( int [ ] arr , int low , int high ) { if ( low < high ) {
int [ ] piv ; piv = partition ( arr , low , high ) ; dualPivotQuickSort ( arr , low , piv [ 0 ] - 1 ) ; dualPivotQuickSort ( arr , piv [ 0 ] + 1 , piv [ 1 ] - 1 ) ; dualPivotQuickSort ( arr , piv [ 1 ] + 1 , high ) ; } } static int [ ] partition ( int [ ] arr , int low , int high ) { if ( arr [ low ] > arr [ high ] ) swap ( arr , low , high ) ;
int j = low + 1 ; int g = high - 1 , k = low + 1 , p = arr [ low ] , q = arr [ high ] ; while ( k <= g ) {
if ( arr [ k ] < p ) { swap ( arr , k , j ) ; j ++ ; }
else if ( arr [ k ] >= q ) { while ( arr [ g ] > q && k < g ) g -- ; swap ( arr , k , g ) ; g -- ; if ( arr [ k ] < p ) { swap ( arr , k , j ) ; j ++ ; } } k ++ ; } j -- ; g ++ ;
swap ( arr , low , j ) ; swap ( arr , high , g ) ;
return new int [ ] { j , g } ; }
public static void main ( String [ ] args ) { int [ ] arr = { 24 , 8 , 42 , 75 , 29 , 77 , 38 , 57 } ; dualPivotQuickSort ( arr , 0 , 7 ) ; System . out . print ( " Sorted ▁ array : ▁ " ) ; for ( int i = 0 ; i < 8 ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; System . out . println ( ) ; } }
import java . util . * ; class GFG { static class pair { int first , second ; public pair ( int first , int second ) { this . first = first ; this . second = second ; } }
static void constGraphWithCon ( int N , int K ) {
int Max = ( ( N - 1 ) * ( N - 2 ) ) / 2 ;
if ( K > Max ) { System . out . print ( - 1 + "NEW_LINE"); return ; }
Vector < pair > ans = new Vector < > ( ) ;
for ( int i = 1 ; i < N ; i ++ ) { for ( int j = i + 1 ; j <= N ; j ++ ) { ans . add ( new pair ( i , j ) ) ; } }
for ( int i = 0 ; i < ( N - 1 ) + Max - K ; i ++ ) { System . out . print ( ans . get ( i ) . first + " ▁ " + ans . get ( i ) . second + "NEW_LINE"); } }
public static void main ( String [ ] args ) { int N = 5 , K = 3 ; constGraphWithCon ( N , K ) ; } }
import java . util . * ; class GFG {
static void findArray ( int N , int K ) {
if ( N == 1 ) { System . out . print ( K + " ▁ " ) ; return ; } if ( N == 2 ) { System . out . print ( 0 + " ▁ " + K ) ; return ; }
int P = N - 2 ; int Q = N - 1 ;
int VAL = 0 ;
for ( int i = 1 ; i <= ( N - 3 ) ; i ++ ) { System . out . print ( i + " ▁ " ) ;
VAL ^= i ; } if ( VAL == K ) { System . out . print ( P + " ▁ " + Q + " ▁ " + ( P ^ Q ) ) ; } else { System . out . print ( 0 + " ▁ " + P + " ▁ " + ( P ^ K ^ VAL ) ) ; } }
public static void main ( String [ ] args ) { int N = 4 , X = 6 ;
findArray ( N , X ) ; } }
class GFG {
static int countDigitSum ( int N , int K ) {
int l = ( int ) Math . pow ( 10 , N - 1 ) , r = ( int ) Math . pow ( 10 , N ) - 1 ; int count = 0 ; for ( int i = l ; i <= r ; i ++ ) { int num = i ;
int digits [ ] = new int [ N ] ; for ( int j = N - 1 ; j >= 0 ; j -- ) { digits [ j ] = num % 10 ; num /= 10 ; } int sum = 0 , flag = 0 ;
for ( int j = 0 ; j < K ; j ++ ) sum += digits [ j ] ;
for ( int j = 1 ; j < N - K + 1 ; j ++ ) { int curr_sum = 0 ; for ( int m = j ; m < j + K ; m ++ ) { curr_sum += digits [ m ] ; }
if ( sum != curr_sum ) { flag = 1 ; break ; } }
if ( flag == 0 ) { count ++ ; } } return count ; }
public static void main ( String [ ] args ) {
int N = 2 , K = 1 ;
System . out . print ( countDigitSum ( N , K ) ) ; } }
class GFG {
public static void convert ( String s ) {
int num = 0 ; int n = s . length ( ) ;
for ( int i = 0 ; i < n ; i ++ )
num = num * 10 + ( ( int ) s . charAt ( i ) - 48 ) ;
System . out . print ( num ) ; }
public static void main ( String [ ] args ) {
String s = "123" ;
convert ( s ) ; } }
static boolean arithmeticThree ( int set [ ] , int n ) {
for ( int j = 1 ; j < n - 1 ; j ++ ) {
int i = j - 1 , k = j + 1 ;
while ( i >= 0 && k <= n - 1 ) { if ( set [ i ] + set [ k ] == 2 * set [ j ] ) return true ; ( set [ i ] + set [ k ] < 2 * set [ j ] ) ? k ++ : i -- ; } } return false ; }
class GFG {
static int maxSumIS ( int arr [ ] , int n ) { int i , j , max = 0 ; int msis [ ] = new int [ n ] ;
for ( i = 0 ; i < n ; i ++ ) msis [ i ] = arr [ i ] ;
for ( i = 1 ; i < n ; i ++ ) for ( j = 0 ; j < i ; j ++ ) if ( arr [ i ] > arr [ j ] && msis [ i ] < msis [ j ] + arr [ i ] ) msis [ i ] = msis [ j ] + arr [ i ] ;
for ( i = 0 ; i < n ; i ++ ) if ( max < msis [ i ] ) max = msis [ i ] ; return max ; }
public static void main ( String args [ ] ) { int arr [ ] = new int [ ] { 1 , 101 , 2 , 3 , 100 , 4 , 5 } ; int n = arr . length ; System . out . println ( " Sum ▁ of ▁ maximum ▁ sum ▁ " + " increasing ▁ subsequence ▁ is ▁ " + maxSumIS ( arr , n ) ) ; } }
import java . util . * ; class GFG {
static int internalSearch ( String needle , int row , int col , String hay [ ] , int row_max , int col_max , int xx ) { int found = 0 ; if ( row >= 0 && row <= row_max && col >= 0 && col <= col_max && xx < needle . length ( ) && needle . charAt ( xx ) == hay [ row ] . charAt ( col ) ) { char match = needle . charAt ( xx ) ; xx += 1 ; hay [ row ] = hay [ row ] . substring ( 0 , col ) + "0" + hay [ row ] . substring ( col + 1 ) ; if ( xx == needle . length ( ) ) { found = 1 ; } else {
found += internalSearch ( needle , row , col + 1 , hay , row_max , col_max , xx ) ; found += internalSearch ( needle , row , col - 1 , hay , row_max , col_max , xx ) ; found += internalSearch ( needle , row + 1 , col , hay , row_max , col_max , xx ) ; found += internalSearch ( needle , row - 1 , col , hay , row_max , col_max , xx ) ; } hay [ row ] = hay [ row ] . substring ( 0 , col ) + match + hay [ row ] . substring ( col + 1 ) ; } return found ; }
static int searchString ( String needle , int row , int col , String str [ ] , int row_count , int col_count ) { int found = 0 ; int r , c ; for ( r = 0 ; r < row_count ; ++ r ) { for ( c = 0 ; c < col_count ; ++ c ) { found += internalSearch ( needle , r , c , str , row_count - 1 , col_count - 1 , 0 ) ; } } return found ; }
public static void main ( String args [ ] ) { String needle = " MAGIC " ; String input [ ] = { " BBABBM " , " CBMBBA " , " IBABBG " , " GOZBBI " , " ABBBBC " , " MCIGAM " } ; String str [ ] = new String [ input . length ] ; int i ; for ( i = 0 ; i < input . length ; ++ i ) { str [ i ] = input [ i ] ; } System . out . println ( " count : ▁ " + searchString ( needle , 0 , 0 , str , str . length , str [ 0 ] . length ( ) ) ) ; } }
class GFG {
public static boolean isBalanced ( String exp ) {
boolean flag = true ; int count = 0 ;
for ( int i = 0 ; i < exp . length ( ) ; i ++ ) { if ( exp . charAt ( i ) == ' ( ' ) { count ++ ; } else {
count -- ; } if ( count < 0 ) {
flag = false ; break ; } }
if ( count != 0 ) { flag = false ; } return flag ; }
public static void main ( String [ ] args ) { String exp1 = " ( ( ( ) ) ) ( ) ( ) " ; if ( isBalanced ( exp1 ) ) System . out . println ( " Balanced " ) ; else System . out . println ( " Not ▁ Balanced " ) ; String exp2 = " ( ) ) ( ( ( ) ) " ; if ( isBalanced ( exp2 ) ) System . out . println ( " Balanced " ) ; else System . out . println ( " Not ▁ Balanced " ) ; } }
class GFG {
static void reverse ( char str [ ] , int start , int end ) {
char temp ; while ( start <= end ) {
temp = str [ start ] ; str [ start ] = str [ end ] ; str [ end ] = temp ; start ++ ; end -- ; } }
static void reverseletter ( char str [ ] , int start , int end ) { int wstart , wend ; for ( wstart = wend = start ; wend < end ; wend ++ ) { if ( str [ wend ] == ' ▁ ' ) { continue ; }
while ( wend <= end && str [ wend ] != ' ▁ ' ) { wend ++ ; } wend -- ;
reverse ( str , wstart , wend ) ; } }
public static void main ( String [ ] args ) { char str [ ] = " Ashish ▁ Yadav ▁ Abhishek ▁ Rajput ▁ Sunil ▁ Pundir " . toCharArray ( ) ; reverseletter ( str , 0 , str . length - 1 ) ; System . out . printf ( " % s " , String . valueOf ( str ) ) ; } }
import java . util . * ; class GFG { static boolean have_same_frequency ( int [ ] freq , int k ) { for ( int i = 0 ; i < 26 ; i ++ ) { if ( freq [ i ] != 0 && freq [ i ] != k ) { return false ; } } return true ; } static int count_substrings ( String s , int k ) { int count = 0 ; int distinct = 0 ; boolean [ ] have = new boolean [ 26 ] ; Arrays . fill ( have , false ) ; for ( int i = 0 ; i < s . length ( ) ; i ++ ) { have [ ( ( int ) ( s . charAt ( i ) - ' a ' ) ) ] = true ; } for ( int i = 0 ; i < 26 ; i ++ ) { if ( have [ i ] ) { distinct ++ ; } } for ( int length = 1 ; length <= distinct ; length ++ ) { int window_length = length * k ; int [ ] freq = new int [ 26 ] ; Arrays . fill ( freq , 0 ) ; int window_start = 0 ; int window_end = window_start + window_length - 1 ; for ( int i = window_start ; i <= Math . min ( window_end , s . length ( ) - 1 ) ; i ++ ) { freq [ ( ( int ) ( s . charAt ( i ) - ' a ' ) ) ] ++ ; } while ( window_end < s . length ( ) ) { if ( have_same_frequency ( freq , k ) ) { count ++ ; } freq [ ( ( int ) ( s . charAt ( window_start ) - ' a ' ) ) ] -- ; window_start ++ ; window_end ++ ; if ( window_end < s . length ( ) ) { freq [ ( ( int ) ( s . charAt ( window_end ) - ' a ' ) ) ] ++ ; } } } return count ; } public static void main ( String [ ] args ) { String s = " aabbcc " ; int k = 2 ; System . out . println ( count_substrings ( s , k ) ) ; s = " aabbc " ; k = 2 ; System . out . println ( count_substrings ( s , k ) ) ; } }
public class Test { static int x = 32 ;
static String toggleCase ( char [ ] a ) { for ( int i = 0 ; i < a . length ; i ++ ) {
a [ i ] ^= 32 ; } return new String ( a ) ; }
public static void main ( String [ ] args ) { String str = " CheRrY " ; System . out . print ( " Toggle ▁ case : ▁ " ) ; str = toggleCase ( str . toCharArray ( ) ) ; System . out . println ( str ) ; System . out . print ( " Original ▁ string : ▁ " ) ; str = toggleCase ( str . toCharArray ( ) ) ; System . out . println ( str ) ; } }
import java . io . * ; import java . util . * ; class GFG { static int NO_OF_CHARS = 256 ;
static boolean areAnagram ( char str1 [ ] , char str2 [ ] ) {
int count1 [ ] = new int [ NO_OF_CHARS ] ; Arrays . fill ( count1 , 0 ) ; int count2 [ ] = new int [ NO_OF_CHARS ] ; Arrays . fill ( count2 , 0 ) ; int i ;
for ( i = 0 ; i < str1 . length && i < str2 . length ; i ++ ) { count1 [ str1 [ i ] ] ++ ; count2 [ str2 [ i ] ] ++ ; }
if ( str1 . length != str2 . length ) return false ;
for ( i = 0 ; i < NO_OF_CHARS ; i ++ ) if ( count1 [ i ] != count2 [ i ] ) return false ; return true ; }
public static void main ( String args [ ] ) { char str1 [ ] = ( " geeksforgeeks " ) . toCharArray ( ) ; char str2 [ ] = ( " forgeeksgeeks " ) . toCharArray ( ) ;
if ( areAnagram ( str1 , str2 ) ) System . out . println ( " The ▁ two ▁ strings ▁ are " + " anagram ▁ of ▁ each ▁ other " ) ; else System . out . println ( " The ▁ two ▁ strings ▁ are ▁ not " + " ▁ anagram ▁ of ▁ each ▁ other " ) ; } }
class GFG {
static int heptacontagonNum ( int n ) { return ( 68 * n * n - 66 * n ) / 2 ; }
public static void main ( String [ ] args ) { int N = 3 ; System . out . println ( "3rd ▁ heptacontagon ▁ Number ▁ is ▁ = ▁ " + heptacontagonNum ( N ) ) ; } }
public class GFG {
static void isEqualFactors ( int N ) { if ( ( N % 2 == 0 ) && ( N % 4 != 0 ) ) System . out . println ( " YES " ) ; else System . out . println ( " NO " ) ; }
public static void main ( String args [ ] ) { int N = 10 ; isEqualFactors ( N ) ; N = 125 ; isEqualFactors ( N ) ; } }
class GFG {
static boolean checkDivisibility ( int n , int digit ) {
return ( digit != 0 && n % digit == 0 ) ; }
static boolean isAllDigitsDivide ( int n ) { int temp = n ; while ( temp > 0 ) {
int digit = temp % 10 ; if ( ! ( checkDivisibility ( n , digit ) ) ) return false ; temp /= 10 ; } return true ; }
static boolean isAllDigitsDistinct ( int n ) {
boolean arr [ ] = new boolean [ 10 ] ;
while ( n > 0 ) {
int digit = n % 10 ;
if ( arr [ digit ] ) return false ;
arr [ digit ] = true ;
n = n / 10 ; } return true ; }
static boolean isLynchBell ( int n ) { return isAllDigitsDivide ( n ) && isAllDigitsDistinct ( n ) ; }
public static void main ( String [ ] args ) {
int N = 12 ;
if ( isLynchBell ( N ) ) System . out . print ( " Yes " ) ; else System . out . print ( " No " ) ; } }
class GFG {
static int maximumAND ( int L , int R ) { return R ; }
public static void main ( String [ ] args ) { int l = 3 ; int r = 7 ; System . out . print ( maximumAND ( l , r ) ) ; } }
import java . util . * ; import java . io . * ; class GFG {
static double findAverageOfCube ( int n ) {
double sum = 0 ;
for ( int i = 1 ; i <= n ; i ++ ) { sum += i * i * i ; }
return sum / n ; }
public static void main ( String [ ] args ) {
int n = 3 ;
System . out . print ( findAverageOfCube ( n ) ) ; } }
class GFG {
static boolean isPower ( int N , int K ) {
int res1 = ( int ) ( Math . log ( N ) / Math . log ( K ) ) ; double res2 = Math . log ( N ) / Math . log ( K ) ;
return ( res1 == res2 ) ; }
public static void main ( String [ ] args ) { int N = 8 ; int K = 2 ; if ( isPower ( N , K ) ) { System . out . println ( " Yes " ) ; } else { System . out . println ( " No " ) ; } } }
class GFG {
static float y ( float x ) { return ( 1 / ( 1 + x ) ) ; }
static float BooleRule ( float a , float b ) {
int n = 4 ; int h ;
h = ( int ) ( ( b - a ) / n ) ; float sum = 0 ;
float bl = ( 7 * y ( a ) + 32 * y ( a + h ) + 12 * y ( a + 2 * h ) + 32 * y ( a + 3 * h ) + 7 * y ( a + 4 * h ) ) * 2 * h / 45 ; sum = sum + bl ; return sum ; }
public static void main ( String [ ] args ) { System . out . printf ( ( " f ( x ) ▁ = ▁ % .4f " ) , BooleRule ( 0 , 4 ) ) ; } }
import java . util . * ; class GFG {
static float y ( float x ) { float num = 1 ; float denom = ( float ) 1.0 + x * x ; return num / denom ; }
static float WeedleRule ( float a , float b ) {
float h = ( b - a ) / 6 ;
float sum = 0 ;
sum = sum + ( ( ( 3 * h ) / 10 ) * ( y ( a ) + y ( a + 2 * h ) + 5 * y ( a + h ) + 6 * y ( a + 3 * h ) + y ( a + 4 * h ) + 5 * y ( a + 5 * h ) + y ( a + 6 * h ) ) ) ;
return sum ; }
public static void main ( String args [ ] ) {
float a = 0 , b = 6 ;
float num = WeedleRule ( a , b ) ; System . out . format ( " f ( x ) ▁ = ▁ " + " % .6f " , num ) ; } }
class GFG {
static double dydx ( double x , double y ) { return ( x + y - 2 ) ; }
static double rungeKutta ( double x0 , double y0 , double x , double h ) {
int n = ( int ) ( ( x - x0 ) / h ) ; double k1 , k2 ;
double y = y0 ; for ( int i = 1 ; i <= n ; i ++ ) {
k1 = h * dydx ( x0 , y ) ; k2 = h * dydx ( x0 + 0.5 * h , y + 0.5 * k1 ) ;
y = y + ( 1.0 / 6.0 ) * ( k1 + 2 * k2 ) ;
x0 = x0 + h ; } return y ; }
public static void main ( String [ ] args ) { double x0 = 0 , y = 1 , x = 2 , h = 0.2 ; System . out . println ( rungeKutta ( x0 , y , x , h ) ) ; } }
import java . lang . * ; class GFG {
public static double per ( double a , double b ) { return ( a + b ) ; }
public static double area ( double s ) { return ( s / 2 ) ; }
public static void main ( String [ ] args ) { double a = 7 , b = 8 , s = 10 ; System . out . println ( per ( a , b ) ) ; System . out . println ( area ( s ) ) ; } }
import java . lang . * ; class GFG { static double PI = 3.14159265 ;
public static double area_leaf ( double a ) { return ( a * a * ( PI / 2 - 1 ) ) ; }
public static void main ( String [ ] args ) { double a = 7 ; System . out . println ( area_leaf ( a ) ) ; } }
import java . lang . * ; class GFG { static double PI = 3.14159265 ;
public static double length_rope ( double r ) { return ( ( 2 * PI * r ) + 6 * r ) ; }
public static void main ( String [ ] args ) { double r = 7 ; System . out . println ( length_rope ( r ) ) ; } }
import java . lang . * ; class GFG { static double PI = 3.14159265 ;
public static double area_inscribed ( double P , double B , double H ) { return ( ( P + B - H ) * ( P + B - H ) * ( PI / 4 ) ) ; }
public static void main ( String [ ] args ) { double P = 3 , B = 4 , H = 5 ; System . out . println ( area_inscribed ( P , B , H ) ) ; } }
import java . lang . * ; class GFG { static double PI = 3.14159265 ;
public static double area_cicumscribed ( double c ) { return ( c * c * ( PI / 4 ) ) ; }
public static void main ( String [ ] args ) { double c = 8.0 ; System . out . println ( area_cicumscribed ( c ) ) ; } }
import java . lang . * ; class GFG { static double PI = 3.14159265 ;
public static double area_inscribed ( double a ) { return ( a * a * ( PI / 12 ) ) ; }
public static double perm_inscribed ( double a ) { return ( PI * ( a / Math . sqrt ( 3 ) ) ) ; }
public static void main ( String [ ] args ) { double a = 6.0 ; System . out . println ( " Area ▁ of ▁ inscribed ▁ circle ▁ is ▁ : " + area_inscribed ( a ) ) ; System . out . println ( " Perimeter of inscribed circle is : " + perm_inscribed ( a ) ) ; } }
import java . io . * ; class GFG {
static float area ( float r ) {
return ( float ) ( ( 0.5 ) * ( 3.14 ) * ( r * r ) ) ; }
static float perimeter ( float r ) {
return ( float ) ( ( 3.14 ) * ( r ) ) ; }
public static void main ( String [ ] args ) {
float r = 10 ;
System . out . println ( " The ▁ Area ▁ of ▁ Semicircle : ▁ " + area ( r ) ) ;
System . out . println ( " The ▁ Perimeter ▁ of ▁ Semicircle : " + + perimeter ( r ) ) ; } }
import java . io . * ; class GFG {
static void equation_plane ( float x1 , float y1 , float z1 , float x2 , float y2 , float z2 , float x3 , float y3 , float z3 ) { float a1 = x2 - x1 ; float b1 = y2 - y1 ; float c1 = z2 - z1 ; float a2 = x3 - x1 ; float b2 = y3 - y1 ; float c2 = z3 - z1 ; float a = b1 * c2 - b2 * c1 ; float b = a2 * c1 - a1 * c2 ; float c = a1 * b2 - b1 * a2 ; float d = ( - a * x1 - b * y1 - c * z1 ) ; System . out . println ( " equation ▁ of ▁ plane ▁ is ▁ " + a + " ▁ x ▁ + ▁ " + b + " ▁ y ▁ + ▁ " + c + " ▁ z ▁ + ▁ " + d + " ▁ = ▁ 0 . " ) ; }
public static void main ( String [ ] args ) { float x1 = - 1 ; float y1 = 2 ; float z1 = 1 ; float x2 = 0 ; float y2 = - 3 ; float z2 = 2 ; float x3 = 1 ; float y3 = 1 ; float z3 = - 4 ; equation_plane ( x1 , y1 , z1 , x2 , y2 , z2 , x3 , y3 , z3 ) ; } }
import java . io . * ; class GFG {
static void shortest_distance ( float x1 , float y1 , float a , float b , float c ) { double d = Math . abs ( ( ( a * x1 + b * y1 + c ) ) / ( Math . sqrt ( a * a + b * b ) ) ) ; System . out . println ( " Perpendicular ▁ " + " distance ▁ is ▁ " + d ) ; return ; }
public static void main ( String [ ] args ) { float x1 = 5 ; float y1 = 6 ; float a = - 2 ; float b = 3 ; float c = 4 ; shortest_distance ( x1 , y1 , a , b , c ) ; } }
import java . util . * ; class solution {
static void octant ( float x , float y , float z ) { if ( x >= 0 && y >= 0 && z >= 0 ) System . out . println ( " Point ▁ lies ▁ in ▁ 1st ▁ octant " ) ; else if ( x < 0 && y >= 0 && z >= 0 ) System . out . println ( " Point ▁ lies ▁ in ▁ 2nd ▁ octant " ) ; else if ( x < 0 && y < 0 && z >= 0 ) System . out . println ( " Point ▁ lies ▁ in ▁ 3rd ▁ octant " ) ; else if ( x >= 0 && y < 0 && z >= 0 ) System . out . println ( " Point ▁ lies ▁ in ▁ 4th ▁ octant " ) ; else if ( x >= 0 && y >= 0 && z < 0 ) System . out . println ( " Point ▁ lies ▁ in ▁ 5th ▁ octant " ) ; else if ( x < 0 && y >= 0 && z < 0 ) System . out . println ( " Point ▁ lies ▁ in ▁ 6th ▁ octant " ) ; else if ( x < 0 && y < 0 && z < 0 ) System . out . println ( " Point ▁ lies ▁ in ▁ 7th ▁ octant " ) ; else if ( x >= 0 && y < 0 && z < 0 ) System . out . println ( " Point ▁ lies ▁ in ▁ 8th ▁ octant " ) ; }
public static void main ( String args [ ] ) { float x = 2 , y = 3 , z = 4 ; octant ( x , y , z ) ; x = - 4 ; y = 2 ; z = - 8 ; octant ( x , y , z ) ; x = - 6 ; y = - 2 ; z = 8 ; octant ( x , y , z ) ; } }
import java . io . * ; class GFG { static double maxArea ( double a , double b , double c , double d ) {
double semiperimeter = ( a + b + c + d ) / 2 ;
return Math . sqrt ( ( semiperimeter - a ) * ( semiperimeter - b ) * ( semiperimeter - c ) * ( semiperimeter - d ) ) ; }
public static void main ( String [ ] args ) { double a = 1 , b = 2 , c = 1 , d = 2 ; System . out . println ( maxArea ( a , b , c , d ) ) ; } }
class GFG {
public static void addAP ( int A [ ] , int Q , int [ ] [ ] operations ) {
for ( int j = 0 ; j < 2 ; ++ j ) { int L = operations [ j ] [ 0 ] , R = operations [ j ] [ 1 ] , a = operations [ j ] [ 2 ] , d = operations [ j ] [ 3 ] ; int curr = a ;
for ( int i = L - 1 ; i < R ; i ++ ) {
A [ i ] += curr ;
curr += d ; } }
for ( int i = 0 ; i < 4 ; ++ i ) System . out . print ( A [ i ] + " ▁ " ) ; }
public static void main ( String args [ ] ) { int A [ ] = { 5 , 4 , 2 , 8 } ; int Q = 2 ; int query [ ] [ ] = { { 1 , 2 , 1 , 3 } , { 1 , 4 , 4 , 1 } } ;
addAP ( A , Q , query ) ; } }
class GFG {
static int N = 5 ;
static class Matrix { int [ ] A ; int size ; } ;
static void Set ( Matrix mat , int i , int j , int x ) { if ( i >= j ) mat . A [ i * ( i - 1 ) / 2 + j - 1 ] = x ; }
static int Get ( Matrix mat , int i , int j ) { if ( i >= j ) { return mat . A [ i * ( i - 1 ) / 2 + j - 1 ] ; } else { return 0 ; } }
static void Display ( Matrix mat ) { int i , j ;
for ( i = 1 ; i <= mat . size ; i ++ ) { for ( j = 1 ; j <= mat . size ; j ++ ) { if ( i >= j ) { System . out . printf ( " % d ▁ " , mat . A [ i * ( i - 1 ) / 2 + j - 1 ] ) ; } else { System . out . printf ( "0 ▁ " ) ; } } System . out . printf ( "NEW_LINE"); } }
static Matrix createMat ( int Mat [ ] [ ] ) {
Matrix mat = new Matrix ( ) ;
mat . size = N ; mat . A = new int [ ( mat . size * ( mat . size + 1 ) ) / 2 ] ; int i , j ;
for ( i = 1 ; i <= mat . size ; i ++ ) { for ( j = 1 ; j <= mat . size ; j ++ ) { Set ( mat , i , j , Mat [ i - 1 ] [ j - 1 ] ) ; } }
return mat ; }
public static void main ( String [ ] args ) { int Mat [ ] [ ] = { { 1 , 0 , 0 , 0 , 0 } , { 1 , 2 , 0 , 0 , 0 } , { 1 , 2 , 3 , 0 , 0 } , { 1 , 2 , 3 , 4 , 0 } , { 1 , 2 , 3 , 4 , 5 } } ;
Matrix mat = createMat ( Mat ) ;
Display ( mat ) ; } }
class GFG { static int log_a_to_base_b ( int a , int b ) { return ( int ) ( Math . log ( a ) / Math . log ( b ) ) ; }
public static void main ( String [ ] args ) { int a = 3 ; int b = 2 ; System . out . println ( log_a_to_base_b ( a , b ) ) ; a = 256 ; b = 4 ; System . out . println ( log_a_to_base_b ( a , b ) ) ; } }
class GFG {
static int log_a_to_base_b ( int a , int b ) { int rslt = ( a > b - 1 ) ? 1 + log_a_to_base_b ( a / b , b ) : 0 ; return rslt ; }
public static void main ( String [ ] args ) { int a = 3 ; int b = 2 ; System . out . println ( log_a_to_base_b ( a , b ) ) ; a = 256 ; b = 4 ; System . out . println ( log_a_to_base_b ( a , b ) ) ; } }
class GFG {
static int maximum ( int x , int y ) { return ( ( x + y + Math . abs ( x - y ) ) / 2 ) ; }
static int minimum ( int x , int y ) { return ( ( x + y - Math . abs ( x - y ) ) / 2 ) ; }
public static void main ( String [ ] args ) { int x = 99 , y = 18 ;
System . out . println ( " Maximum : ▁ " + maximum ( x , y ) ) ;
System . out . println ( " Minimum : ▁ " + minimum ( x , y ) ) ; } }
import java . text . * ; class GFG {
static double p = 1 , f = 1 ; static double e ( int x , int n ) { double r ;
if ( n == 0 ) return 1 ;
r = e ( x , n - 1 ) ;
p = p * x ;
f = f * n ; return ( r + p / f ) ; }
public static void main ( String [ ] args ) { int x = 4 , n = 15 ; DecimalFormat df = new DecimalFormat ( "0 . # # # # # # " ) ; System . out . println ( df . format ( e ( x , n ) ) ) ; } }
import java . util . * ; import java . text . DecimalFormat ; class GFG { static void midptellipse ( float rx , float ry , float xc , float yc ) { float dx , dy , d1 , d2 , x , y ; x = 0 ; y = ry ;
d1 = ( ry * ry ) - ( rx * rx * ry ) + ( 0.25f * rx * rx ) ; dx = 2 * ry * ry * x ; dy = 2 * rx * rx * y ; DecimalFormat df = new DecimalFormat ( " # , # # # , # #0.00000" ) ;
while ( dx < dy ) {
System . out . println ( df . format ( ( x + xc ) ) + " , ▁ " + df . format ( ( y + yc ) ) ) ; System . out . println ( df . format ( ( - x + xc ) ) + " , ▁ " + df . format ( ( y + yc ) ) ) ; System . out . println ( df . format ( ( x + xc ) ) + " , ▁ " + df . format ( ( - y + yc ) ) ) ; System . out . println ( df . format ( ( - x + xc ) ) + " , ▁ " + df . format ( ( - y + yc ) ) ) ;
if ( d1 < 0 ) { x ++ ; dx = dx + ( 2 * ry * ry ) ; d1 = d1 + dx + ( ry * ry ) ; } else { x ++ ; y -- ; dx = dx + ( 2 * ry * ry ) ; dy = dy - ( 2 * rx * rx ) ; d1 = d1 + dx - dy + ( ry * ry ) ; } }
d2 = ( ( ry * ry ) * ( ( x + 0.5f ) * ( x + 0.5f ) ) ) + ( ( rx * rx ) * ( ( y - 1 ) * ( y - 1 ) ) ) - ( rx * rx * ry * ry ) ;
while ( y >= 0 ) {
System . out . println ( df . format ( ( x + xc ) ) + " , ▁ " + df . format ( ( y + yc ) ) ) ; System . out . println ( df . format ( ( - x + xc ) ) + " , ▁ " + df . format ( ( y + yc ) ) ) ; System . out . println ( df . format ( ( x + xc ) ) + " , ▁ " + df . format ( ( - y + yc ) ) ) ; System . out . println ( df . format ( ( - x + xc ) ) + " , ▁ " + df . format ( ( - y + yc ) ) ) ;
if ( d2 > 0 ) { y -- ; dy = dy - ( 2 * rx * rx ) ; d2 = d2 + ( rx * rx ) - dy ; } else { y -- ; x ++ ; dx = dx + ( 2 * ry * ry ) ; dy = dy - ( 2 * rx * rx ) ; d2 = d2 + dx - dy + ( rx * rx ) ; } } }
public static void main ( String args [ ] ) {
midptellipse ( 10 , 15 , 50 , 50 ) ; } }
public class Improve {
static void HexToBin ( char hexdec [ ] ) { int i = 0 ; while ( hexdec [ i ] != ' \u0000' ) { switch ( hexdec [ i ] ) { case '0' : System . out . print ( "0000" ) ; break ; case '1' : System . out . print ( "0001" ) ; break ; case '2' : System . out . print ( "0010" ) ; break ; case '3' : System . out . print ( "0011" ) ; break ; case '4' : System . out . print ( "0100" ) ; break ; case '5' : System . out . print ( "0101" ) ; break ; case '6' : System . out . print ( "0110" ) ; break ; case '7' : System . out . print ( "0111" ) ; break ; case '8' : System . out . print ( "1000" ) ; break ; case '9' : System . out . print ( "1001" ) ; break ; case ' A ' : case ' a ' : System . out . print ( "1010" ) ; break ; case ' B ' : case ' b ' : System . out . print ( "1011" ) ; break ; case ' C ' : case ' c ' : System . out . print ( "1100" ) ; break ; case ' D ' : case ' d ' : System . out . print ( "1101" ) ; break ; case ' E ' : case ' e ' : System . out . print ( "1110" ) ; break ; case ' F ' : case ' f ' : System . out . print ( "1111" ) ; break ; default : System . out . print ( " Invalid hexadecimal digit " + hexdec[i]); } i ++ ; } }
public static void main ( String args [ ] ) {
String s = "1AC5" ; char hexdec [ ] = new char [ 100 ] ; hexdec = s . toCharArray ( ) ;
System . out . print ( " Equivalent Binary value is : "); try { HexToBin ( hexdec ) ; } catch ( ArrayIndexOutOfBoundsException e ) { System . out . print ( " " ) ; } } }
import java . io . * ; class GFG { public static void main ( String [ ] args ) { int matrix [ ] [ ] = new int [ 5 ] [ 5 ] , row_index , column_index , x = 0 , size = 5 ;
for ( row_index = 0 ; row_index < size ; row_index ++ ) { for ( column_index = 0 ; column_index < size ; column_index ++ ) { matrix [ row_index ] [ column_index ] = ++ x ; } }
System . out . printf ( "The matrix isNEW_LINE"); for ( row_index = 0 ; row_index < size ; row_index ++ ) { for ( column_index = 0 ; column_index < size ; column_index ++ ) { System . out . printf ( " % d TABSYMBOL " , matrix [ row_index ] [ column_index ] ) ; } System . out . printf ( "NEW_LINE"); }
System . out . printf ( " Elements on Secondary diagonal : "); for ( row_index = 0 ; row_index < size ; row_index ++ ) { for ( column_index = 0 ; column_index < size ; column_index ++ ) {
if ( ( row_index + column_index ) == size - 1 ) System . out . printf ( " % d , ▁ " , matrix [ row_index ] [ column_index ] ) ; } } } }
class GFG { public static void main ( String [ ] args ) { int [ ] [ ] matrix = new int [ 5 ] [ 5 ] ; int row_index , column_index , x = 0 , size = 5 ;
for ( row_index = 0 ; row_index < size ; row_index ++ ) { for ( column_index = 0 ; column_index < size ; column_index ++ ) { matrix [ row_index ] [ column_index ] = ++ x ; } }
System . out . printf ( "The matrix isNEW_LINE"); for ( row_index = 0 ; row_index < size ; row_index ++ ) { for ( column_index = 0 ; column_index < size ; column_index ++ ) { System . out . printf ( " % d TABSYMBOL " , matrix [ row_index ] [ column_index ] ) ; } System . out . printf ( "NEW_LINE"); }
System . out . printf ( " Elements above Secondary " ▁ + ▁ " diagonal are : "); for ( row_index = 0 ; row_index < size ; row_index ++ ) { for ( column_index = 0 ; column_index < size ; column_index ++ ) {
if ( ( row_index + column_index ) < size - 1 ) System . out . printf ( " % d , ▁ " , matrix [ row_index ] [ column_index ] ) ; } } } }
class GFG { public static void main ( String [ ] args ) { int [ ] [ ] matrix = new int [ 5 ] [ 5 ] ; int row_index , column_index , x = 0 , size = 5 ;
for ( row_index = 0 ; row_index < size ; row_index ++ ) { for ( column_index = 0 ; column_index < size ; column_index ++ ) { matrix [ row_index ] [ column_index ] = ++ x ; } }
System . out . printf ( "The matrix isNEW_LINE"); for ( row_index = 0 ; row_index < size ; row_index ++ ) { for ( column_index = 0 ; column_index < size ; column_index ++ ) { System . out . printf ( " % d TABSYMBOL " , matrix [ row_index ] [ column_index ] ) ; } System . out . printf ( "NEW_LINE"); }
System . out . printf ( " Corner Elements are : "); for ( row_index = 0 ; row_index < size ; row_index ++ ) { for ( column_index = 0 ; column_index < size ; column_index ++ ) {
if ( ( row_index == 0 row_index == size - 1 ) && ( column_index == 0 column_index == size - 1 ) ) System . out . printf ( " % d , ▁ " , matrix [ row_index ] [ column_index ] ) ; } } } }
import java . io . * ; import java . lang . Math ; class GFG {
static void distance ( float x1 , float y1 , float z1 , float x2 , float y2 , float z2 ) { double d = Math . pow ( ( Math . pow ( x2 - x1 , 2 ) + Math . pow ( y2 - y1 , 2 ) + Math . pow ( z2 - z1 , 2 ) * 1.0 ) , 0.5 ) ; System . out . println ( " Distance ▁ is ▁ " + d ) ; return ; }
public static void main ( String [ ] args ) { float x1 = 2 ; float y1 = - 5 ; float z1 = 7 ; float x2 = 3 ; float y2 = 4 ; float z2 = 5 ;
distance ( x1 , y1 , z1 , x2 , y2 , z2 ) ; } }
import java . io . * ; class GFG {
static int No_Of_Pairs ( int N ) { int i = 1 ;
while ( ( i * i * i ) + ( 2 * i * i ) + i <= N ) i ++ ; return ( i - 1 ) ; }
static void print_pairs ( int pairs ) { int i = 1 , mul ; for ( i = 1 ; i <= pairs ; i ++ ) { mul = i * ( i + 1 ) ; System . out . println ( " Pair ▁ no . ▁ " + i + " ▁ - - > ▁ ( " + ( mul * i ) + " , ▁ " + mul * ( i + 1 ) + " ) " ) ; } }
public static void main ( String [ ] args ) { int N = 500 , pairs , mul , i = 1 ; pairs = No_Of_Pairs ( N ) ; System . out . println ( " No . ▁ of ▁ pairs ▁ = ▁ " + pairs ) ; print_pairs ( pairs ) ; } }
class GFG {
static double findArea ( double d ) { return ( d * d ) / 2 ; }
public static void main ( String [ ] args ) { double d = 10 ; System . out . println ( findArea ( d ) ) ; } }
import java . io . * ; public class GFG {
static float AvgofSquareN ( int n ) { float sum = 0 ; for ( int i = 1 ; i <= n ; i ++ ) sum += ( i * i ) ; return sum / n ; }
static public void main ( String [ ] args ) { int n = 2 ; System . out . println ( AvgofSquareN ( n ) ) ; } }
import java . io . * ; class MathSeries {
static double Series ( double x , int n ) { double sum = 1 , term = 1 , fct , j , y = 2 , m ;
int i ; for ( i = 1 ; i < n ; i ++ ) { fct = 1 ; for ( j = 1 ; j <= y ; j ++ ) { fct = fct * j ; } term = term * ( - 1 ) ; m = Math . pow ( x , y ) / fct ; m = m * term ; sum = sum + m ; y += 2 ; } return sum ; }
public static void main ( String [ ] args ) { double x = 3 ; int n = 4 ; System . out . println ( Math . round ( Series ( x , n ) * 10000.0 ) / 10000.0 ) ; } }
import java . io . * ; import java . util . * ; class GFG {
static long maxPrimeFactors ( long n ) {
long maxPrime = - 1 ;
while ( n % 2 == 0 ) { maxPrime = 2 ;
n >>= 1 ; }
while ( n % 3 == 0 ) { maxPrime = 3 ; n = n / 3 ; }
for ( int i = 5 ; i <= Math . sqrt ( n ) ; i += 6 ) { while ( n % i == 0 ) { maxPrime = i ; n = n / i ; } while ( n % ( i + 2 ) == 0 ) { maxPrime = i + 2 ; n = n / ( i + 2 ) ; } }
if ( n > 4 ) maxPrime = n ; return maxPrime ; }
public static void main ( String [ ] args ) { Long n = 15l ; System . out . println ( maxPrimeFactors ( n ) ) ; n = 25698751364526l ; System . out . println ( maxPrimeFactors ( n ) ) ; } }
class GFG {
static double sum ( int x , int n ) { double i , total = 1.0 , multi = x ; for ( i = 1 ; i <= n ; i ++ ) { total = total + multi / i ; multi = multi * x ; } return total ; }
public static void main ( String [ ] args ) { int x = 2 ; int n = 5 ; System . out . printf ( " % .2f " , sum ( x , n ) ) ; } }
class GFG {
static int chiliagonNum ( int n ) { return ( 998 * n * n - 996 * n ) / 2 ; }
public static void main ( String [ ] args ) { int n = 3 ; System . out . println ( "3rd ▁ chiliagon ▁ Number ▁ is ▁ = ▁ " + chiliagonNum ( n ) ) ; } }
import java . util . * ; class GFG {
static int pentacontagonNum ( int n ) { return ( 48 * n * n - 46 * n ) / 2 ; }
public static void main ( String [ ] args ) { int n = 3 ; System . out . println ( "3rd ▁ pentacontagon ▁ Number ▁ is ▁ = ▁ " + pentacontagonNum ( n ) ) ; } }
import java . util . * ; class GFG {
static int lastElement ( int [ ] arr ) {
PriorityQueue < Integer > pq = new PriorityQueue < > ( ( a , b ) -> b - a ) ; for ( int i = 0 ; i < arr . length ; i ++ ) pq . add ( arr [ i ] ) ;
int m1 , m2 ;
while ( ! pq . isEmpty ( ) ) {
if ( pq . size ( ) == 1 ) {
return pq . poll ( ) ; } m1 = pq . poll ( ) ; m2 = pq . poll ( ) ;
if ( m1 != m2 ) pq . add ( m1 - m2 ) ; }
return 0 ; }
public static void main ( String [ ] args ) { int [ ] arr = new int [ ] { 2 , 7 , 4 , 1 , 8 , 1 , 1 } ; System . out . println ( lastElement ( arr ) ) ; } }
class GFG {
static int countDigit ( double n ) { return ( ( int ) Math . floor ( Math . log10 ( n ) + 1 ) ) ; }
public static void main ( String [ ] args ) { double N = 80 ; System . out . println ( countDigit ( N ) ) ; } }
class GFG {
static double sum ( int x , int n ) { double i , total = 1.0 , multi = x ;
System . out . print ( "1 ▁ " ) ;
for ( i = 1 ; i < n ; i ++ ) { total = total + multi ; System . out . print ( multi ) ; System . out . print ( " ▁ " ) ; multi = multi * x ; } System . out . println ( ) ; return total ; }
public static void main ( String [ ] args ) { int x = 2 ; int n = 5 ; System . out . printf ( " % .2f " , sum ( x , n ) ) ; } }
class Main {
public static void main ( String [ ] args ) { int N = 43 ; int ans = findRemainder ( N ) ; System . out . println ( ans ) ; }
public static int findRemainder ( int n ) {
int x = n & 3 ;
return x ; } }
import java . util . * ; class GFG {
static void triangular_series ( int n ) { int i , j = 1 , k = 1 ;
for ( i = 1 ; i <= n ; i ++ ) { System . out . printf ( " % d ▁ " , k ) ;
j = j + 1 ;
k = k + j ; } }
public static void main ( String [ ] args ) { int n = 5 ; triangular_series ( n ) ; } }
import java . util . * ; class GFG { static int countDigit ( long n ) { if ( n / 10 == 0 ) return 1 ; return 1 + countDigit ( n / 10 ) ; }
public static void main ( String [ ] args ) { long n = 345289467 ; System . out . print ( " Number ▁ of ▁ digits ▁ : ▁ " + countDigit ( n ) ) ; } }
class GFG { public static void main ( String [ ] args ) {
int x = 1234 ;
if ( x % 9 == 1 ) System . out . printf ( " Magic ▁ Number " ) ; else System . out . printf ( " Not ▁ a ▁ Magic ▁ Number " ) ; } }
class GFG { static int MAX = 100 ; public static void main ( String [ ] args ) {
long [ ] arr = new long [ MAX ] ; arr [ 0 ] = 0 ; arr [ 1 ] = 1 ; for ( int i = 2 ; i < MAX ; i ++ ) arr [ i ] = arr [ i - 1 ] + arr [ i - 2 ] ; System . out . print ( " Fibonacci ▁ numbers ▁ divisible ▁ by ▁ " + "their indexes are :NEW_LINE"); for ( int i = 1 ; i < MAX ; i ++ ) if ( arr [ i ] % i == 0 ) System . out . print ( i + " ▁ " ) ; } }
import java . io . * ; import java . util . * ; class GFG { public static int findMaxValue ( ) { int res = 2 ; long fact = 2 ; while ( true ) {
if ( fact < 0 ) break ; res ++ ; fact = fact * res ; } return res - 1 ; }
public static void main ( String [ ] args ) { System . out . println ( " Maximum ▁ value ▁ of " + " ▁ integer ▁ " + findMaxValue ( ) ) ; } }
import java . util . * ; import java . lang . * ; import java . io . * ; class KDigitSquare {
public static long firstkdigits ( int n , int k ) {
double product = n * Math . log10 ( n ) ;
double decimal_part = product - Math . floor ( product ) ;
decimal_part = Math . pow ( 10 , decimal_part ) ;
double digits = Math . pow ( 10 , k - 1 ) , i = 0 ; return ( ( long ) ( decimal_part * digits ) ) ; }
public static void main ( String [ ] args ) { int n = 1450 ; int k = 6 ; System . out . println ( firstkdigits ( n , k ) ) ; } }
class GFG {
static long moduloMultiplication ( long a , long b , long mod ) {
a %= mod ; while ( b > 0 ) {
if ( ( b & 1 ) > 0 ) { res = ( res + a ) % mod ; }
a = ( 2 * a ) % mod ;
} return res ; }
public static void main ( String [ ] args ) { long a = 10123465234878998L ; long b = 65746311545646431L ; long m = 10005412336548794L ; System . out . print ( moduloMultiplication ( a , b , m ) ) ; } }
import java . util . * ; class GFG { static int canBeSumofConsec ( int n ) {
n = 2 * n ;
return ( ( n & ( n - 1 ) ) != 0 ) ? 1 : 0 ; } public static void main ( String [ ] args ) { int n = 10 ; System . out . print ( canBeSumofConsec ( n ) + "NEW_LINE"); } }
import java . io . * ; import static java . lang . Math . * ; class Quadratic {
static void findRoots ( int a , int b , int c ) {
if ( a == 0 ) { System . out . println ( " Invalid " ) ; return ; } int d = b * b - 4 * a * c ; double sqrt_val = sqrt ( abs ( d ) ) ; if ( d > 0 ) { System . out . println ( "Roots are real and different NEW_LINE"); System . out . println ( ( double ) ( - b + sqrt_val ) / ( 2 * a ) + "NEW_LINE" + ( double ) ( - b - sqrt_val ) / ( 2 * a ) ) ; } else if ( d == 0 ) { System . out . println ( "Roots are real and same NEW_LINE"); System . out . println ( - ( double ) b / ( 2 * a ) + "NEW_LINE" + - ( double ) b / ( 2 * a ) ) ; }
{ System . out . println ( "Roots are complex NEW_LINE"); System . out . println ( - ( double ) b / ( 2 * a ) + " ▁ + ▁ i " + sqrt_val + "NEW_LINE" + - ( double ) b / ( 2 * a ) + " ▁ - ▁ i " + sqrt_val ) ; } }
public static void main ( String args [ ] ) { int a = 1 , b = - 7 , c = 12 ;
findRoots ( a , b , c ) ; } }
import java . io . * ; class GFG {
static int val ( char c ) { if ( c >= '0' && c <= '9' ) return ( int ) c - '0' ; else return ( int ) c - ' A ' + 10 ; }
static int toDeci ( String str , int base ) { int len = str . length ( ) ;
int power = 1 ;
int num = 0 ; int i ;
for ( i = len - 1 ; i >= 0 ; i -- ) {
if ( val ( str . charAt ( i ) ) >= base ) { System . out . println ( " Invalid ▁ Number " ) ; return - 1 ; } num += val ( str . charAt ( i ) ) * power ; power = power * base ; } return num ; }
public static void main ( String [ ] args ) { String str = "11A " ; int base = 16 ; System . out . println ( " Decimal ▁ equivalent ▁ of ▁ " + str + " ▁ in ▁ base ▁ " + base + " ▁ is ▁ " + " ▁ " + toDeci ( str , base ) ) ; } }
class GFG {
static int seriesSum ( int calculated , int current , int N ) { int i , cur = 1 ;
if ( current == N + 1 ) return 0 ;
for ( i = calculated ; i < calculated + current ; i ++ ) cur *= i ;
return cur + seriesSum ( i , current + 1 , N ) ; }
public static void main ( String [ ] args ) {
int N = 5 ;
System . out . println ( seriesSum ( 1 , 1 , N ) ) ; } }
import java . io . * ; class GFG {
public static int N = 30 ;
public static int [ ] fib = new int [ N ] ;
public static int largestFiboLessOrEqual ( int n ) {
fib [ 0 ] = 1 ;
fib [ 1 ] = 2 ;
int i ; for ( i = 2 ; fib [ i - 1 ] <= n ; i ++ ) { fib [ i ] = fib [ i - 1 ] + fib [ i - 2 ] ; }
return ( i - 2 ) ; }
public static String fibonacciEncoding ( int n ) { int index = largestFiboLessOrEqual ( n ) ;
char [ ] codeword = new char [ index + 3 ] ;
int i = index ; while ( n > 0 ) {
codeword [ i ] = '1' ;
n = n - fib [ i ] ;
i = i - 1 ;
while ( i >= 0 && fib [ i ] > n ) { codeword [ i ] = '0' ; i = i - 1 ; } }
codeword [ index + 1 ] = '1' ; codeword [ index + 2 ] = ' \0' ; String string = new String ( codeword ) ;
return string ; }
public static void main ( String [ ] args ) { int n = 143 ; System . out . println ( " Fibonacci ▁ code ▁ word ▁ for ▁ " + n + " ▁ is ▁ " + fibonacciEncoding ( n ) ) ; } }
import java . util . * ; class GFG {
static int countSquares ( int m , int n ) {
if ( n < m ) { int temp = m ; m = n ; n = temp ; }
return n * ( n + 1 ) * ( 3 * m - n + 1 ) / 6 ; }
public static void main ( String [ ] args ) { int m = 4 ; int n = 3 ; System . out . print ( " Count ▁ of ▁ squares ▁ is ▁ " + countSquares ( m , n ) ) ; } }
static void simpleSieve ( int limit ) {
boolean [ ] mark = new boolean [ limit ] ; Arrays . fill ( mark , true ) ;
for ( int p = 2 ; p * p < limit ; p ++ ) {
if ( mark [ p ] == true ) {
for ( int i = p * p ; i < limit ; i += p ) mark [ i ] = false ; } }
for ( int p = 2 ; p < limit ; p ++ ) if ( mark [ p ] == true ) System . out . print ( p + " ▁ " ) ; }
class GFG {
static int modInverse ( int a , int m ) { int m0 = m ; int y = 0 , x = 1 ; if ( m == 1 ) return 0 ; while ( a > 1 ) {
int q = a / m ; int t = m ;
m = a % m ; a = t ; t = y ;
y = x - q * y ; x = t ; }
if ( x < 0 ) x += m0 ; return x ; }
public static void main ( String args [ ] ) { int a = 3 , m = 11 ;
System . out . println ( " Modular ▁ multiplicative ▁ " + " inverse ▁ is ▁ " + modInverse ( a , m ) ) ; } }
import java . io . * ; class GFG {
static int gcd ( int a , int b ) { if ( a == 0 ) return b ; return gcd ( b % a , a ) ; }
static int phi ( int n ) { int result = 1 ; for ( int i = 2 ; i < n ; i ++ ) if ( gcd ( i , n ) == 1 ) result ++ ; return result ; }
public static void main ( String [ ] args ) { int n ; for ( n = 1 ; n <= 10 ; n ++ ) System . out . println ( " phi ( " + n + " ) ▁ = ▁ " + phi ( n ) ) ; } }
import java . io . * ; class GFG { static int phi ( int n ) {
for ( int p = 2 ; p * p <= n ; ++ p ) {
if ( n % p == 0 ) {
while ( n % p == 0 ) n /= p ; result *= ( 1.0 - ( 1.0 / ( float ) p ) ) ; } }
if ( n > 1 ) result *= ( 1.0 - ( 1.0 / ( float ) n ) ) ; return ( int ) result ; }
public static void main ( String args [ ] ) { int n ; for ( n = 1 ; n <= 10 ; n ++ ) System . out . println ( " phi ( " + n + " ) ▁ = ▁ " + phi ( n ) ) ; } }
class Test {
static void printFibonacciNumbers ( int n ) { int f1 = 0 , f2 = 1 , i ; if ( n < 1 ) return ; System . out . print ( f1 + " ▁ " ) ; for ( i = 1 ; i < n ; i ++ ) { System . out . print ( f2 + " ▁ " ) ; int next = f1 + f2 ; f1 = f2 ; f2 = next ; } }
public static void main ( String [ ] args ) { printFibonacciNumbers ( 7 ) ; } }
class Test {
static int gcd ( int a , int b ) { if ( a == 0 ) return b ; return gcd ( b % a , a ) ; }
static int lcm ( int a , int b ) { return ( a / gcd ( a , b ) ) * b ; }
public static void main ( String [ ] args ) { int a = 15 , b = 20 ; System . out . println ( " LCM ▁ of ▁ " + a + " ▁ and ▁ " + b + " ▁ is ▁ " + lcm ( a , b ) ) ; } }
class GFG {
static void convert_to_words ( char [ ] num ) {
if ( len == 0 ) { System . out . println ( " empty ▁ string " ) ; return ; } if ( len > 4 ) { System . out . println ( " Length ▁ more ▁ than ▁ 4 ▁ is ▁ not ▁ supported " ) ; return ; }
String [ ] single_digits = new String [ ] { " zero " , " one " , " two " , " three " , " four " , " five " , " six " , " seven " , " eight " , " nine " } ;
String [ ] two_digits = new String [ ] { " " , " ten " , " eleven " , " twelve " , " thirteen " , " fourteen " , " fifteen " , " sixteen " , " seventeen " , " eighteen " , " nineteen " } ;
String [ ] tens_multiple = new String [ ] { " " , " " , " twenty " , " thirty " , " forty " , " fifty " , " sixty " , " seventy " , " eighty " , " ninety " } ; String [ ] tens_power = new String [ ] { " hundred " , " thousand " } ;
System . out . print ( String . valueOf ( num ) + " : ▁ " ) ;
if ( len == 1 ) { System . out . println ( single_digits [ num [ 0 ] - '0' ] ) ; return ; }
int x = 0 ; while ( x < num . length ) {
if ( len >= 3 ) { if ( num [ x ] - '0' != 0 ) { System . out . print ( single_digits [ num [ x ] - '0' ] + " ▁ " ) ; System . out . print ( tens_power [ len - 3 ] + " ▁ " ) ;
} -- len ; }
else {
if ( num [ x ] - '0' == 1 ) { int sum = num [ x ] - '0' + num [ x + 1 ] - '0' ; System . out . println ( two_digits [ sum ] ) ; return ; }
else if ( num [ x ] - '0' == 2 && num [ x + 1 ] - '0' == 0 ) { System . out . println ( " twenty " ) ; return ; }
else { int i = ( num [ x ] - '0' ) ; if ( i > 0 ) System . out . print ( tens_multiple [ i ] + " ▁ " ) ; else System . out . print ( " " ) ; ++ x ; if ( num [ x ] - '0' != 0 ) System . out . println ( single_digits [ num [ x ] - '0' ] ) ; } } ++ x ; } }
public static void main ( String [ ] args ) { convert_to_words ( "9923" . toCharArray ( ) ) ; convert_to_words ( "523" . toCharArray ( ) ) ; convert_to_words ( "89" . toCharArray ( ) ) ; convert_to_words ( "8" . toCharArray ( ) ) ; } }
class GFG { static int MAX = 11 ; static boolean isMultipleof5 ( int n ) { char str [ ] = new char [ MAX ] ; int len = str . length ;
if ( str [ len - 1 ] == '5' str [ len - 1 ] == '0' ) return true ; return false ; }
public static void main ( String [ ] args ) { int n = 19 ; if ( isMultipleof5 ( n ) == true ) System . out . println ( n + " ▁ is ▁ multiple ▁ " + " of ▁ 5" ) ; else System . out . println ( n + " ▁ is ▁ not ▁ a ▁ " + " multiple ▁ of ▁ 5" ) ; } }
class GFG {
static int toggleBit ( int n , int k ) { return ( n ^ ( 1 << ( k - 1 ) ) ) ; }
public static void main ( String [ ] args ) { int n = 5 , k = 2 ; System . out . printf ( "%dNEW_LINE", toggleBit(n, k)); } }
class GFG {
static int clearBit ( int n , int k ) { return ( n & ( ~ ( 1 << ( k - 1 ) ) ) ) ; }
public static void main ( String [ ] args ) { int n = 5 , k = 1 ; System . out . println ( clearBit ( n , k ) ) ; } }
import java . io . * ; class GFG { static int add ( int x , int y ) { int keep = ( x & y ) << 1 ; int res = x ^ y ;
if ( keep == 0 ) return res ; return add ( keep , res ) ; }
public static void main ( String [ ] args ) { System . out . println ( add ( 15 , 38 ) ) ; } }
import java . io . * ; class GFG { static int countBits ( int number ) {
return ( int ) ( Math . log ( number ) / Math . log ( 2 ) + 1 ) ; }
public static void main ( String [ ] args ) { int num = 65 ; System . out . println ( countBits ( num ) ) ; } }
class GFG { static int INT_SIZE = 32 ;
static int constructNthNumber ( int group_no , int aux_num , int op ) { int a [ ] = new int [ INT_SIZE ] ; int num = 0 , len_f ; int i = 0 ;
if ( op == 2 ) {
len_f = 2 * group_no ;
a [ len_f - 1 ] = a [ 0 ] = 1 ;
while ( aux_num > 0 ) {
a [ group_no + i ] = a [ group_no - 1 - i ] = aux_num & 1 ; aux_num = aux_num >> 1 ; i ++ ; } }
else if ( op == 0 ) {
len_f = 2 * group_no + 1 ;
a [ len_f - 1 ] = a [ 0 ] = 1 ; a [ group_no ] = 0 ;
while ( aux_num > 0 ) {
a [ group_no + 1 + i ] = a [ group_no - 1 - i ] = aux_num & 1 ; aux_num = aux_num >> 1 ; i ++ ; } }
{
len_f = 2 * group_no + 1 ;
a [ len_f - 1 ] = a [ 0 ] = 1 ; a [ group_no ] = 1 ;
while ( aux_num > 0 ) {
a [ group_no + 1 + i ] = a [ group_no - 1 - i ] = aux_num & 1 ; aux_num = aux_num >> 1 ; i ++ ; } }
for ( i = 0 ; i < len_f ; i ++ ) num += ( 1 << i ) * a [ i ] ; return num ; }
static int getNthNumber ( int n ) { int group_no = 0 , group_offset ; int count_upto_group = 0 , count_temp = 1 ; int op , aux_num ;
while ( count_temp < n ) { group_no ++ ;
count_upto_group = count_temp ; count_temp += 3 * ( 1 << ( group_no - 1 ) ) ; }
group_offset = n - count_upto_group - 1 ;
if ( ( group_offset + 1 ) <= ( 1 << ( group_no - 1 ) ) ) {
aux_num = group_offset ; } else { if ( ( ( group_offset + 1 ) - ( 1 << ( group_no - 1 ) ) ) % 2 == 1 )
else
aux_num = ( ( group_offset ) - ( 1 << ( group_no - 1 ) ) ) / 2 ; } return constructNthNumber ( group_no , aux_num , op ) ; }
public static void main ( String [ ] args ) { int n = 9 ;
System . out . printf ( " % d " , getNthNumber ( n ) ) ; } }
import java . io . * ; class GFG {
static int swapBits ( int n , int p1 , int p2 ) {
int bit1 = ( n >> p1 ) & 1 ;
int bit2 = ( n >> p2 ) & 1 ;
int x = ( bit1 ^ bit2 ) ;
x = ( x << p1 ) | ( x << p2 ) ;
int result = n ^ x ; return result ; }
public static void main ( String [ ] args ) { int res = swapBits ( 28 , 0 , 3 ) ; System . out . println ( " Result ▁ = ▁ " + res ) ; } }
static int invert ( int x ) { if ( x == 1 ) return 2 ; else return 1 ; }
static int invertSub ( int x ) { return ( 3 - x ) ; }
import java . io . * ; import java . util . * ; import java . lang . * ; class GFG {
static int firstNonRepeating ( String str ) { int NO_OF_CHARS = 256 ;
int arr [ ] = new int [ NO_OF_CHARS ] ; for ( int i = 0 ; i < NO_OF_CHARS ; i ++ ) arr [ i ] = - 1 ;
for ( int i = 0 ; i < str . length ( ) ; i ++ ) { if ( arr [ str . charAt ( i ) ] == - 1 ) arr [ str . charAt ( i ) ] = i ; else arr [ str . charAt ( i ) ] = - 2 ; } int res = Integer . MAX_VALUE ; for ( int i = 0 ; i < NO_OF_CHARS ; i ++ )
if ( arr [ i ] >= 0 ) res = Math . min ( res , arr [ i ] ) ; return res ; }
public static void main ( String args [ ] ) { String str = " geeksforgeeks " ; int index = firstNonRepeating ( str ) ; if ( index == Integer . MAX_VALUE ) System . out . print ( " Either ▁ all ▁ characters ▁ are ▁ " + " repeating ▁ or ▁ string ▁ is ▁ empty " ) ; else System . out . print ( " First ▁ non - repeating ▁ character " + " ▁ is ▁ " + str . charAt ( index ) ) ; } }
import java . io . * ; import java . util . * ; class GFG {
static int triacontagonalNum ( int n ) { return ( 28 * n * n - 26 * n ) / 2 ; }
public static void main ( String [ ] args ) { int n = 3 ; System . out . println ( "3rd ▁ triacontagonal ▁ Number ▁ is ▁ = ▁ " + triacontagonalNum ( n ) ) ; } }
class GFG {
public static int hexacontagonNum ( int n ) { return ( 58 * n * n - 56 * n ) / 2 ; }
public static void main ( String [ ] args ) { int n = 3 ; System . out . println ( "3rd ▁ hexacontagon ▁ Number ▁ is ▁ = ▁ " + hexacontagonNum ( n ) ) ; } }
class GFG {
public static int enneacontagonNum ( int n ) { return ( 88 * n * n - 86 * n ) / 2 ; }
public static void main ( String [ ] args ) { int n = 3 ; System . out . println ( "3rd ▁ enneacontagon ▁ Number ▁ is ▁ = ▁ " + enneacontagonNum ( n ) ) ; } }
class GFG {
public static int triacontakaidigonNum ( int n ) { return ( 30 * n * n - 28 * n ) / 2 ; }
public static void main ( String [ ] args ) { int n = 3 ; System . out . println ( "3rd ▁ triacontakaidigon ▁ Number ▁ is ▁ = ▁ " + triacontakaidigonNum ( n ) ) ; } }
class GFG {
public static int IcosihexagonalNum ( int n ) { return ( 24 * n * n - 22 * n ) / 2 ; }
public static void main ( String [ ] args ) { int n = 3 ; System . out . println ( "3rd ▁ Icosihexagonal ▁ Number ▁ is ▁ = ▁ " + IcosihexagonalNum ( n ) ) ; } }
class GFG {
public static int icosikaioctagonalNum ( int n ) { return ( 26 * n * n - 24 * n ) / 2 ; }
public static void main ( String [ ] args ) { int n = 3 ; System . out . println ( "3rd ▁ icosikaioctagonal ▁ Number ▁ is ▁ = ▁ " + icosikaioctagonalNum ( n ) ) ; } }
import java . util . * ; class GFG {
static int octacontagonNum ( int n ) { return ( 78 * n * n - 76 * n ) / 2 ; }
public static void main ( String args [ ] ) { int n = 3 ; System . out . print ( "3rd ▁ octacontagon ▁ Number ▁ is ▁ = ▁ " + octacontagonNum ( n ) ) ; } }
import java . util . * ; class GFG {
static int hectagonNum ( int n ) { return ( 98 * n * n - 96 * n ) / 2 ; }
public static void main ( String args [ ] ) { int n = 3 ; System . out . print ( "3rd ▁ hectagon ▁ Number ▁ is ▁ = ▁ " + hectagonNum ( n ) ) ; } }
import java . util . * ; class GFG {
static int tetracontagonNum ( int n ) { return ( 38 * n * n - 36 * n ) / 2 ; }
public static void main ( String [ ] args ) { int n = 3 ; System . out . println ( "3rd ▁ tetracontagon ▁ Number ▁ is ▁ = ▁ " + tetracontagonNum ( n ) ) ; } }
class GFG {
static int binarySearch ( int arr [ ] , int N , int X ) {
int start = 0 ;
int end = N ; while ( start <= end ) {
int mid = start + ( end - start ) / 2 ;
if ( X == arr [ mid ] ) {
return mid ; }
else if ( X < arr [ mid ] ) {
start = mid + 1 ; } else {
end = mid - 1 ; } }
return - 1 ; }
public static void main ( String [ ] args ) { int arr [ ] = { 5 , 4 , 3 , 2 , 1 } ; int N = arr . length ; int X = 5 ; System . out . println ( binarySearch ( arr , N , X ) ) ; } }
import java . io . * ; class PancakeSort {
static void flip ( int arr [ ] , int i ) { int temp , start = 0 ; while ( start < i ) { temp = arr [ start ] ; arr [ start ] = arr [ i ] ; arr [ i ] = temp ; start ++ ; i -- ; } }
static int findMax ( int arr [ ] , int n ) { int mi , i ; for ( mi = 0 , i = 0 ; i < n ; ++ i ) if ( arr [ i ] > arr [ mi ] ) mi = i ; return mi ; }
static int pancakeSort ( int arr [ ] , int n ) {
for ( int curr_size = n ; curr_size > 1 ; -- curr_size ) {
int mi = findMax ( arr , curr_size ) ;
if ( mi != curr_size - 1 ) {
flip ( arr , mi ) ;
flip ( arr , curr_size - 1 ) ; } } return 0 ; }
static void printArray ( int arr [ ] , int arr_size ) { for ( int i = 0 ; i < arr_size ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; System . out . println ( " " ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 23 , 10 , 20 , 11 , 12 , 6 , 7 } ; int n = arr . length ; pancakeSort ( arr , n ) ; System . out . println ( " Sorted ▁ Array : ▁ " ) ; printArray ( arr , n ) ; } }
