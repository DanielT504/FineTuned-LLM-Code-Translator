def count_setbit ( N ) : NEW_LINE
result = 0 NEW_LINE
for i in range ( 32 ) : NEW_LINE
if ( ( 1 << i ) & N ) : NEW_LINE
result = result + 1 NEW_LINE print ( result ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 43 NEW_LINE count_setbit ( N ) NEW_LINE DEDENT
import math NEW_LINE
def isPowerOfTwo ( n ) : NEW_LINE INDENT return ( math . ceil ( math . log ( n ) // math . log ( 2 ) ) == math . floor ( math . log ( n ) // math . log ( 2 ) ) ) ; NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 8 NEW_LINE if isPowerOfTwo ( N ) : NEW_LINE INDENT print ( ' Yes ' ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( ' No ' ) NEW_LINE DEDENT DEDENT
def search ( pat , txt ) : NEW_LINE INDENT M = len ( pat ) NEW_LINE N = len ( txt ) NEW_LINE i = 0 NEW_LINE while i <= N - M : NEW_LINE DEDENT
for j in xrange ( M ) : NEW_LINE INDENT if txt [ i + j ] != pat [ j ] : NEW_LINE INDENT break NEW_LINE DEDENT j += 1 NEW_LINE DEDENT
print " Pattern ▁ found ▁ at ▁ index ▁ " + str ( i ) NEW_LINE i = i + M NEW_LINE elif j == 0 : NEW_LINE i = i + 1 NEW_LINE else : NEW_LINE
i = i + j NEW_LINE
txt = " ABCEABCDABCEABCD " NEW_LINE pat = " ABCD " NEW_LINE search ( pat , txt ) NEW_LINE
def encrypt ( input_arr ) : NEW_LINE
evenPos = ' @ ' ; oddPos = ' ! ' ; NEW_LINE for i in range ( len ( input_arr ) ) : NEW_LINE
ascii = ord ( input_arr [ i ] ) ; NEW_LINE repeat = ( ascii - 96 ) if ascii >= 97 else ( ascii - 64 ) ; NEW_LINE for j in range ( repeat ) : NEW_LINE
' NEW_LINE INDENT if ( i % 2 == 0 ) : NEW_LINE INDENT print ( oddPos , end = " " ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( evenPos , end = " " ) ; NEW_LINE DEDENT DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT input_arr = [ ' A ' , ' b ' , ' C ' , ' d ' ] ; NEW_LINE DEDENT
encrypt ( input_arr ) ; NEW_LINE
def isPalRec ( st , s , e ) : NEW_LINE
if ( s == e ) : NEW_LINE INDENT return True NEW_LINE DEDENT
if ( st [ s ] != st [ e ] ) : NEW_LINE INDENT return False NEW_LINE DEDENT
if ( s < e + 1 ) : NEW_LINE INDENT return isPalRec ( st , s + 1 , e - 1 ) ; NEW_LINE DEDENT return True NEW_LINE def isPalindrome ( st ) : NEW_LINE n = len ( st ) NEW_LINE
if ( n == 0 ) : NEW_LINE INDENT return True NEW_LINE DEDENT return isPalRec ( st , 0 , n - 1 ) ; NEW_LINE
st = " geeg " NEW_LINE if ( isPalindrome ( st ) ) : NEW_LINE INDENT print " Yes " NEW_LINE DEDENT else : NEW_LINE INDENT print " No " NEW_LINE DEDENT
import sys NEW_LINE def myAtoi ( Str ) : NEW_LINE INDENT sign , base , i = 1 , 0 , 0 NEW_LINE DEDENT
while ( Str [ i ] == ' ▁ ' ) : NEW_LINE INDENT i += 1 NEW_LINE DEDENT
if ( Str [ i ] == ' - ' or Str [ i ] == ' + ' ) : NEW_LINE INDENT sign = 1 - 2 * ( Str [ i ] == ' - ' ) NEW_LINE i += 1 NEW_LINE DEDENT
while ( i < len ( Str ) and Str [ i ] >= '0' and Str [ i ] <= '9' ) : NEW_LINE
if ( base > ( sys . maxsize // 10 ) or ( base == ( sys . maxsize // 10 ) and ( Str [ i ] - '0' ) > 7 ) ) : NEW_LINE INDENT if ( sign == 1 ) : NEW_LINE INDENT return sys . maxsize NEW_LINE DEDENT else : NEW_LINE INDENT return - ( sys . maxsize ) NEW_LINE DEDENT DEDENT base = 10 * base + ( ord ( Str [ i ] ) - ord ( '0' ) ) NEW_LINE i += 1 NEW_LINE return base * sign NEW_LINE
Str = list ( " ▁ - 123" ) NEW_LINE
val = myAtoi ( Str ) NEW_LINE print ( val ) NEW_LINE
def fillUtil ( res , curr , n ) : NEW_LINE
if curr == 0 : NEW_LINE INDENT return True NEW_LINE DEDENT
for i in range ( 2 * n - curr - 1 ) : NEW_LINE
if res [ i ] == 0 and res [ i + curr + 1 ] == 0 : NEW_LINE
' NEW_LINE INDENT res [ i ] = res [ i + curr + 1 ] = curr NEW_LINE DEDENT
if fillUtil ( res , curr - 1 , n ) : NEW_LINE INDENT return True NEW_LINE DEDENT
res [ i ] = 0 NEW_LINE res [ i + curr + 1 ] = 0 NEW_LINE return False NEW_LINE def fill ( n ) : NEW_LINE
res = [ 0 ] * ( 2 * n ) NEW_LINE
if fillUtil ( res , n , n ) : NEW_LINE INDENT for i in range ( 2 * n ) : NEW_LINE INDENT print ( res [ i ] , end = ' ▁ ' ) NEW_LINE DEDENT print ( ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Not ▁ Possible " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT fill ( 7 ) NEW_LINE DEDENT
import math NEW_LINE
def findNumberOfDigits ( n , base ) : NEW_LINE
dig = ( math . floor ( math . log ( n ) / math . log ( base ) ) + 1 ) NEW_LINE
return dig NEW_LINE
def isAllKs ( n , b , k ) : NEW_LINE INDENT len = findNumberOfDigits ( n , b ) NEW_LINE DEDENT
sum = k * ( 1 - pow ( b , len ) ) / ( 1 - b ) NEW_LINE return sum == N NEW_LINE
N = 13 NEW_LINE
B = 3 NEW_LINE
K = 1 NEW_LINE
if ( isAllKs ( N , B , K ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
def CalPeri ( ) : NEW_LINE INDENT s = 5 NEW_LINE Perimeter = 10 * s NEW_LINE print ( " The ▁ Perimeter ▁ of ▁ Decagon ▁ is ▁ : ▁ " , Perimeter ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT CalPeri ( ) ; NEW_LINE DEDENT
import math NEW_LINE
def distance ( a1 , b1 , c1 , a2 , b2 , c2 ) : NEW_LINE INDENT d = ( a1 * a2 + b1 * b2 + c1 * c2 ) NEW_LINE e1 = math . sqrt ( a1 * a1 + b1 * b1 + c1 * c1 ) NEW_LINE e2 = math . sqrt ( a2 * a2 + b2 * b2 + c2 * c2 ) NEW_LINE d = d / ( e1 * e2 ) NEW_LINE A = math . degrees ( math . acos ( d ) ) NEW_LINE print ( " Angle ▁ is " ) , A , ( " degree " ) NEW_LINE DEDENT
a1 = 1 NEW_LINE b1 = 1 NEW_LINE c1 = 2 NEW_LINE d1 = 1 NEW_LINE a2 = 2 NEW_LINE b2 = - 1 NEW_LINE c2 = 1 NEW_LINE d2 = - 4 NEW_LINE distance ( a1 , b1 , c1 , a2 , b2 , c2 ) NEW_LINE
def mirror_point ( a , b , c , d , x1 , y1 , z1 ) : NEW_LINE INDENT k = ( - a * x1 - b * y1 - c * z1 - d ) / float ( ( a * a + b * b + c * c ) ) NEW_LINE x2 = a * k + x1 NEW_LINE y2 = b * k + y1 NEW_LINE z2 = c * k + z1 NEW_LINE x3 = 2 * x2 - x1 NEW_LINE y3 = 2 * y2 - y1 NEW_LINE z3 = 2 * z2 - z1 NEW_LINE print " x3 ▁ = " , x3 , NEW_LINE print " y3 ▁ = " , y3 , NEW_LINE print " z3 ▁ = " , z3 , NEW_LINE DEDENT
a = 1 NEW_LINE b = - 2 NEW_LINE c = 0 NEW_LINE d = 0 NEW_LINE x1 = - 1 NEW_LINE y1 = 3 NEW_LINE z1 = 4 NEW_LINE
mirror_point ( a , b , c , d , x1 , y1 , z1 ) NEW_LINE
class newNode : NEW_LINE
def __init__ ( self , key ) : NEW_LINE INDENT self . data = key NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT
def updatetree ( root ) : NEW_LINE
if ( not root ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT if ( root . left == None and root . right == None ) : NEW_LINE INDENT return root . data NEW_LINE DEDENT
leftsum = updatetree ( root . left ) NEW_LINE rightsum = updatetree ( root . right ) NEW_LINE
root . data += leftsum NEW_LINE
return root . data + rightsum NEW_LINE
def inorder ( node ) : NEW_LINE INDENT if ( node == None ) : NEW_LINE INDENT return NEW_LINE DEDENT inorder ( node . left ) NEW_LINE print ( node . data , end = " ▁ " ) NEW_LINE inorder ( node . right ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE
root = newNode ( 1 ) NEW_LINE root . left = newNode ( 2 ) NEW_LINE root . right = newNode ( 3 ) NEW_LINE root . left . left = newNode ( 4 ) NEW_LINE root . left . right = newNode ( 5 ) NEW_LINE root . right . right = newNode ( 6 ) NEW_LINE updatetree ( root ) NEW_LINE print ( " Inorder ▁ traversal ▁ of ▁ the ▁ modified ▁ tree ▁ is " ) NEW_LINE inorder ( root ) NEW_LINE
def calculateSpan ( price , n , S ) : NEW_LINE
S [ 0 ] = 1 NEW_LINE
for i in range ( 1 , n , 1 ) : NEW_LINE
S [ i ] = 1 NEW_LINE
j = i - 1 NEW_LINE while ( j >= 0 ) and ( price [ i ] >= price [ j ] ) : NEW_LINE INDENT S [ i ] += 1 NEW_LINE j -= 1 NEW_LINE DEDENT
def printArray ( arr , n ) : NEW_LINE INDENT for i in range ( n ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT DEDENT
price = [ 10 , 4 , 5 , 90 , 120 , 80 ] NEW_LINE n = len ( price ) NEW_LINE S = [ None ] * n NEW_LINE
calculateSpan ( price , n , S ) NEW_LINE
printArray ( S , n ) NEW_LINE
def printNGE ( arr ) : NEW_LINE INDENT for i in range ( 0 , len ( arr ) , 1 ) : NEW_LINE INDENT next = - 1 NEW_LINE for j in range ( i + 1 , len ( arr ) , 1 ) : NEW_LINE INDENT if arr [ i ] < arr [ j ] : NEW_LINE INDENT next = arr [ j ] NEW_LINE break NEW_LINE DEDENT DEDENT print ( str ( arr [ i ] ) + " ▁ - - ▁ " + str ( next ) ) NEW_LINE DEDENT DEDENT
arr = [ 11 , 13 , 21 , 3 ] NEW_LINE printNGE ( arr ) NEW_LINE
class newNode : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = self . right = None NEW_LINE DEDENT DEDENT
def mirror ( node ) : NEW_LINE INDENT if ( node == None ) : NEW_LINE INDENT return NEW_LINE DEDENT else : NEW_LINE INDENT temp = node NEW_LINE DEDENT DEDENT
mirror ( node . left ) NEW_LINE mirror ( node . right ) NEW_LINE
temp = node . left NEW_LINE node . left = node . right NEW_LINE node . right = temp NEW_LINE
def inOrder ( node ) : NEW_LINE INDENT if ( node == None ) : NEW_LINE INDENT return NEW_LINE DEDENT inOrder ( node . left ) NEW_LINE print ( node . data , end = " ▁ " ) NEW_LINE inOrder ( node . right ) NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT root = newNode ( 1 ) NEW_LINE root . left = newNode ( 2 ) NEW_LINE root . right = newNode ( 3 ) NEW_LINE root . left . left = newNode ( 4 ) NEW_LINE root . left . right = newNode ( 5 ) NEW_LINE DEDENT
print ( " Inorder ▁ traversal ▁ of ▁ the " , " constructed ▁ tree ▁ is " ) NEW_LINE inOrder ( root ) NEW_LINE
mirror ( root ) NEW_LINE
print ( " Inorder traversal of " , ▁ " the mirror treeis   " ) NEW_LINE inOrder ( root ) NEW_LINE
class newNode : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = self . right = None NEW_LINE DEDENT DEDENT
def IsFoldable ( root ) : NEW_LINE INDENT if ( root == None ) : NEW_LINE INDENT return True NEW_LINE DEDENT return IsFoldableUtil ( root . left , root . right ) NEW_LINE DEDENT
def IsFoldableUtil ( n1 , n2 ) : NEW_LINE
if n1 == None and n2 == None : NEW_LINE INDENT return True NEW_LINE DEDENT
if n1 == None or n2 == None : NEW_LINE INDENT return False NEW_LINE DEDENT
d1 = IsFoldableUtil ( n1 . left , n2 . right ) NEW_LINE d2 = IsFoldableUtil ( n1 . right , n2 . left ) NEW_LINE return d1 and d2 NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
root = newNode ( 1 ) NEW_LINE root . left = newNode ( 2 ) NEW_LINE root . right = newNode ( 3 ) NEW_LINE root . left . right = newNode ( 4 ) NEW_LINE root . right . left = newNode ( 5 ) NEW_LINE if IsFoldable ( root ) : NEW_LINE INDENT print ( " Tree ▁ is ▁ foldable " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Tree ▁ is ▁ not ▁ foldable " ) NEW_LINE DEDENT
def isSumProperty ( node ) : NEW_LINE
left_data = 0 NEW_LINE right_data = 0 NEW_LINE
if ( node == None or ( node . left == None and node . right == None ) ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT else : NEW_LINE
if ( node . left != None ) : NEW_LINE INDENT left_data = node . left . data NEW_LINE DEDENT
if ( node . right != None ) : NEW_LINE INDENT right_data = node . right . data NEW_LINE DEDENT
if ( ( node . data == left_data + right_data ) and isSumProperty ( node . left ) and isSumProperty ( node . right ) ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT else : NEW_LINE INDENT return 0 NEW_LINE DEDENT
class newNode : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT root = newNode ( 10 ) NEW_LINE root . left = newNode ( 8 ) NEW_LINE root . right = newNode ( 2 ) NEW_LINE root . left . left = newNode ( 3 ) NEW_LINE root . left . right = newNode ( 5 ) NEW_LINE root . right . right = newNode ( 2 ) NEW_LINE if ( isSumProperty ( root ) ) : NEW_LINE INDENT print ( " The ▁ given ▁ tree ▁ satisfies ▁ the " , " children ▁ sum ▁ property ▁ " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " The ▁ given ▁ tree ▁ does ▁ not ▁ satisfy " , " the ▁ children ▁ sum ▁ property ▁ " ) NEW_LINE DEDENT DEDENT
def gcd ( a , b ) : NEW_LINE
if ( a == 0 and b == 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT if ( a == 0 ) : NEW_LINE INDENT return b NEW_LINE DEDENT if ( b == 0 ) : NEW_LINE INDENT return a NEW_LINE DEDENT
if ( a == b ) : NEW_LINE INDENT return a NEW_LINE DEDENT
if ( a > b ) : NEW_LINE INDENT return gcd ( a - b , b ) NEW_LINE DEDENT return gcd ( a , b - a ) NEW_LINE
a = 98 NEW_LINE b = 56 NEW_LINE if ( gcd ( a , b ) ) : NEW_LINE INDENT print ( ' GCD ▁ of ' , a , ' and ' , b , ' is ' , gcd ( a , b ) ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( ' not ▁ found ' ) NEW_LINE DEDENT
def msbPos ( n ) : NEW_LINE INDENT pos = 0 NEW_LINE while n != 0 : NEW_LINE INDENT pos += 1 NEW_LINE DEDENT DEDENT
n = n >> 1 NEW_LINE return pos NEW_LINE
def josephify ( n ) : NEW_LINE
position = msbPos ( n ) NEW_LINE
' NEW_LINE INDENT j = 1 << ( position - 1 ) NEW_LINE DEDENT
n = n ^ j NEW_LINE
n = n << 1 NEW_LINE
n = n | 1 NEW_LINE return n NEW_LINE
n = 41 NEW_LINE print ( josephify ( n ) ) NEW_LINE
def pairAndSum ( arr , n ) : NEW_LINE
for i in range ( 0 , 32 ) : NEW_LINE
k = 0 NEW_LINE for j in range ( 0 , n ) : NEW_LINE INDENT if ( ( arr [ j ] & ( 1 << i ) ) ) : NEW_LINE INDENT k = k + 1 NEW_LINE DEDENT DEDENT
ans = ans + ( 1 << i ) * ( k * ( k - 1 ) // 2 ) NEW_LINE return ans NEW_LINE
arr = [ 5 , 10 , 15 ] NEW_LINE n = len ( arr ) NEW_LINE print ( pairAndSum ( arr , n ) ) NEW_LINE
def countSquares ( n ) : NEW_LINE
return ( ( n * ( n + 1 ) / 2 ) * ( 2 * n + 1 ) / 3 ) NEW_LINE
n = 4 NEW_LINE print ( " Count ▁ of ▁ squares ▁ is ▁ " , countSquares ( n ) ) NEW_LINE
def gcd ( a , b ) : NEW_LINE
if ( a == 0 ) : NEW_LINE INDENT return b NEW_LINE DEDENT if ( b == 0 ) : NEW_LINE INDENT return a NEW_LINE DEDENT
if ( a == b ) : NEW_LINE INDENT return a NEW_LINE DEDENT
if ( a > b ) : NEW_LINE INDENT return gcd ( a - b , b ) NEW_LINE DEDENT return gcd ( a , b - a ) NEW_LINE
a = 98 NEW_LINE b = 56 NEW_LINE if ( gcd ( a , b ) ) : NEW_LINE INDENT print ( ' GCD ▁ of ' , a , ' and ' , b , ' is ' , gcd ( a , b ) ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( ' not ▁ found ' ) NEW_LINE DEDENT
from math import ceil , log NEW_LINE
maxsize = 100005 NEW_LINE
xor_tree = [ 0 ] * maxsize NEW_LINE
def construct_Xor_Tree_Util ( current , start , end , x ) : NEW_LINE
if ( start == end ) : NEW_LINE INDENT xor_tree [ x ] = current [ start ] NEW_LINE DEDENT
return NEW_LINE
left = x * 2 + 1 NEW_LINE
right = x * 2 + 2 NEW_LINE
mid = start + ( end - start ) // 2 NEW_LINE
construct_Xor_Tree_Util ( current , start , mid , left ) NEW_LINE construct_Xor_Tree_Util ( current , mid + 1 , end , right ) NEW_LINE
xor_tree [ x ] = ( xor_tree [ left ] ^ xor_tree [ right ] ) NEW_LINE
def construct_Xor_Tree ( arr , n ) : NEW_LINE INDENT construct_Xor_Tree_Util ( arr , 0 , n - 1 , 0 ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE
leaf_nodes = [ 40 , 32 , 12 , 1 , 4 , 3 , 2 , 7 ] NEW_LINE n = len ( leaf_nodes ) NEW_LINE
construct_Xor_Tree ( leaf_nodes , n ) NEW_LINE
x = ( ceil ( log ( n , 2 ) ) ) NEW_LINE
max_size = 2 * pow ( 2 , x ) - 1 NEW_LINE print ( " Nodes ▁ of ▁ the ▁ XOR ▁ Tree : " ) NEW_LINE for i in range ( max_size ) : NEW_LINE INDENT print ( xor_tree [ i ] , end = " ▁ " ) NEW_LINE DEDENT
root = 0 NEW_LINE
print ( " Root :   " , xor_tree [ root ] ) NEW_LINE
def swapBits ( n , p1 , p2 ) : NEW_LINE
INDENT n ^= 1 << p1 NEW_LINE n ^= 1 << p2 NEW_LINE return n NEW_LINE DEDENT
print ( " Result ▁ = " , swapBits ( 28 , 0 , 3 ) ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def isSibling ( root , a , b ) : NEW_LINE
if root is None : NEW_LINE INDENT return 0 NEW_LINE DEDENT return ( ( root . left == a and root . right == b ) or ( root . left == b and root . right == a ) or isSibling ( root . left , a , b ) or isSibling ( root . right , a , b ) ) NEW_LINE
def level ( root , ptr , lev ) : NEW_LINE
if root is None : NEW_LINE INDENT return 0 NEW_LINE DEDENT if root == ptr : NEW_LINE INDENT return lev NEW_LINE DEDENT
l = level ( root . left , ptr , lev + 1 ) NEW_LINE if l != 0 : NEW_LINE INDENT return l NEW_LINE DEDENT
return level ( root . right , ptr , lev + 1 ) NEW_LINE
def isCousin ( root , a , b ) : NEW_LINE
if ( ( level ( root , a , 1 ) == level ( root , b , 1 ) ) and not ( isSibling ( root , a , b ) ) ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT else : NEW_LINE INDENT return 0 NEW_LINE DEDENT
root = Node ( 1 ) NEW_LINE root . left = Node ( 2 ) NEW_LINE root . right = Node ( 3 ) NEW_LINE root . left . left = Node ( 4 ) NEW_LINE root . left . right = Node ( 5 ) NEW_LINE root . left . right . right = Node ( 15 ) NEW_LINE root . right . left = Node ( 6 ) NEW_LINE root . right . right = Node ( 7 ) NEW_LINE root . right . left . right = Node ( 8 ) NEW_LINE node1 = root . left . right NEW_LINE node2 = root . right . right NEW_LINE print " Yes " if isCousin ( root , node1 , node2 ) == 1 else " No " NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def checkUtil ( root , level ) : NEW_LINE
if root is None : NEW_LINE INDENT return True NEW_LINE DEDENT
if root . left is None and root . right is None : NEW_LINE
if check . leafLevel == 0 : NEW_LINE
check . leafLevel = level NEW_LINE return True NEW_LINE
return level == check . leafLevel NEW_LINE
return ( checkUtil ( root . left , level + 1 ) and checkUtil ( root . right , level + 1 ) ) NEW_LINE
def check ( root ) : NEW_LINE INDENT level = 0 NEW_LINE check . leafLevel = 0 NEW_LINE return ( checkUtil ( root , level ) ) NEW_LINE DEDENT
root = Node ( 12 ) NEW_LINE root . left = Node ( 5 ) NEW_LINE root . left . left = Node ( 3 ) NEW_LINE root . left . right = Node ( 9 ) NEW_LINE root . left . left . left = Node ( 1 ) NEW_LINE root . left . right . left = Node ( 2 ) NEW_LINE if ( check ( root ) ) : NEW_LINE INDENT print " Leaves ▁ are ▁ at ▁ same ▁ level " NEW_LINE DEDENT else : NEW_LINE INDENT print " Leaves ▁ are ▁ not ▁ at ▁ same ▁ level " NEW_LINE DEDENT
class Node : NEW_LINE INDENT def __init__ ( self , key ) : NEW_LINE INDENT self . key = key NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def isFullTree ( root ) : NEW_LINE
if root is None : NEW_LINE INDENT return True NEW_LINE DEDENT
if root . left is None and root . right is None : NEW_LINE INDENT return True NEW_LINE DEDENT
if root . left is not None and root . right is not None : NEW_LINE INDENT return ( isFullTree ( root . left ) and isFullTree ( root . right ) ) NEW_LINE DEDENT
return False NEW_LINE
root = Node ( 10 ) ; NEW_LINE root . left = Node ( 20 ) ; NEW_LINE root . right = Node ( 30 ) ; NEW_LINE root . left . right = Node ( 40 ) ; NEW_LINE root . left . left = Node ( 50 ) ; NEW_LINE root . right . left = Node ( 60 ) ; NEW_LINE root . right . right = Node ( 70 ) ; NEW_LINE root . left . left . left = Node ( 80 ) ; NEW_LINE root . left . left . right = Node ( 90 ) ; NEW_LINE root . left . right . left = Node ( 80 ) ; NEW_LINE root . left . right . right = Node ( 90 ) ; NEW_LINE root . right . left . left = Node ( 80 ) ; NEW_LINE root . right . left . right = Node ( 90 ) ; NEW_LINE root . right . right . left = Node ( 80 ) ; NEW_LINE root . right . right . right = Node ( 90 ) ; NEW_LINE if isFullTree ( root ) : NEW_LINE INDENT print " The ▁ Binary ▁ tree ▁ is ▁ full " NEW_LINE DEDENT else : NEW_LINE INDENT print " Binary ▁ tree ▁ is ▁ not ▁ full " NEW_LINE DEDENT
def printAlter ( arr , N ) : NEW_LINE
for currIndex in range ( 0 , N , 2 ) : NEW_LINE
print ( arr [ currIndex ] , end = " ▁ " ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 1 , 2 , 3 , 4 , 5 ] NEW_LINE N = len ( arr ) NEW_LINE printAlter ( arr , N ) NEW_LINE DEDENT
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def identicalTrees ( a , b ) : NEW_LINE
if a is None and b is None : NEW_LINE INDENT return True NEW_LINE DEDENT
if a is not None and b is not None : NEW_LINE INDENT return ( ( a . data == b . data ) and identicalTrees ( a . left , b . left ) and identicalTrees ( a . right , b . right ) ) NEW_LINE DEDENT
return False NEW_LINE
root1 = Node ( 1 ) NEW_LINE root2 = Node ( 1 ) NEW_LINE root1 . left = Node ( 2 ) NEW_LINE root1 . right = Node ( 3 ) NEW_LINE root1 . left . left = Node ( 4 ) NEW_LINE root1 . left . right = Node ( 5 ) NEW_LINE root2 . left = Node ( 2 ) NEW_LINE root2 . right = Node ( 3 ) NEW_LINE root2 . left . left = Node ( 4 ) NEW_LINE root2 . left . right = Node ( 5 ) NEW_LINE if identicalTrees ( root1 , root2 ) : NEW_LINE INDENT print " Both ▁ trees ▁ are ▁ identical " NEW_LINE DEDENT else : NEW_LINE INDENT print " Trees ▁ are ▁ not ▁ identical " NEW_LINE DEDENT
class newNode : NEW_LINE INDENT def __init__ ( self , item ) : NEW_LINE INDENT self . data = item NEW_LINE self . left = self . right = None NEW_LINE DEDENT DEDENT
def getLevel ( root , node , level ) : NEW_LINE
if ( root == None ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT if ( root == node ) : NEW_LINE INDENT return level NEW_LINE DEDENT
downlevel = getLevel ( root . left , node , level + 1 ) NEW_LINE if ( downlevel != 0 ) : NEW_LINE INDENT return downlevel NEW_LINE DEDENT
return getLevel ( root . right , node , level + 1 ) NEW_LINE
def printGivenLevel ( root , node , level ) : NEW_LINE
if ( root == None or level < 2 ) : NEW_LINE INDENT return NEW_LINE DEDENT
if ( level == 2 ) : NEW_LINE INDENT if ( root . left == node or root . right == node ) : NEW_LINE INDENT return NEW_LINE DEDENT if ( root . left ) : NEW_LINE INDENT print ( root . left . data , end = " ▁ " ) NEW_LINE DEDENT if ( root . right ) : NEW_LINE INDENT print ( root . right . data , end = " ▁ " ) NEW_LINE DEDENT DEDENT
elif ( level > 2 ) : NEW_LINE INDENT printGivenLevel ( root . left , node , level - 1 ) NEW_LINE printGivenLevel ( root . right , node , level - 1 ) NEW_LINE DEDENT
def printCousins ( root , node ) : NEW_LINE
level = getLevel ( root , node , 1 ) NEW_LINE
printGivenLevel ( root , node , level ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT root = newNode ( 1 ) NEW_LINE root . left = newNode ( 2 ) NEW_LINE root . right = newNode ( 3 ) NEW_LINE root . left . left = newNode ( 4 ) NEW_LINE root . left . right = newNode ( 5 ) NEW_LINE root . left . right . right = newNode ( 15 ) NEW_LINE root . right . left = newNode ( 6 ) NEW_LINE root . right . right = newNode ( 7 ) NEW_LINE root . right . left . right = newNode ( 8 ) NEW_LINE printCousins ( root , root . left . right ) NEW_LINE DEDENT
def leftRotate ( arr , d , n ) : NEW_LINE INDENT leftRotateRec ( arr , 0 , d , n ) ; NEW_LINE DEDENT def leftRotateRec ( arr , i , d , n ) : NEW_LINE
if ( d == 0 or d == n ) : NEW_LINE INDENT return ; NEW_LINE DEDENT
if ( n - d == d ) : NEW_LINE INDENT swap ( arr , i , n - d + i , d ) ; NEW_LINE return ; NEW_LINE DEDENT
if ( d < n - d ) : NEW_LINE INDENT swap ( arr , i , n - d + i , d ) ; NEW_LINE leftRotateRec ( arr , i , d , n - d ) ; NEW_LINE DEDENT
else : NEW_LINE INDENT swap ( arr , i , d , n - d ) ; NEW_LINE leftRotateRec ( arr , n - d + i , 2 * d - n , d ) ; NEW_LINE DEDENT
def printArray ( arr , size ) : NEW_LINE INDENT for i in range ( size ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) ; NEW_LINE DEDENT print ( ) ; NEW_LINE DEDENT
def swap ( arr , fi , si , d ) : NEW_LINE INDENT for i in range ( d ) : NEW_LINE INDENT temp = arr [ fi + i ] ; NEW_LINE arr [ fi + i ] = arr [ si + i ] ; NEW_LINE arr [ si + i ] = temp ; NEW_LINE DEDENT DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 1 , 2 , 3 , 4 , 5 , 6 , 7 ] ; NEW_LINE leftRotate ( arr , 2 , 7 ) ; NEW_LINE printArray ( arr , 7 ) ; NEW_LINE DEDENT
def leftRotate ( arr , d , n ) : NEW_LINE INDENT if ( d == 0 or d == n ) : NEW_LINE INDENT return ; NEW_LINE DEDENT i = d NEW_LINE j = n - d NEW_LINE while ( i != j ) : NEW_LINE DEDENT
if ( i < j ) : NEW_LINE INDENT swap ( arr , d - i , d + j - i , i ) NEW_LINE j -= i NEW_LINE DEDENT
else : NEW_LINE INDENT swap ( arr , d - i , d , j ) NEW_LINE i -= j NEW_LINE DEDENT
swap ( arr , d - i , d , i ) NEW_LINE
def rotate ( arr , n ) : NEW_LINE INDENT i = 0 NEW_LINE j = n - 1 NEW_LINE while i != j : NEW_LINE arr [ i ] , arr [ j ] = arr [ j ] , arr [ i ] NEW_LINE i = i + 1 NEW_LINE pass NEW_LINE DEDENT
arr = [ 1 , 2 , 3 , 4 , 5 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Given ▁ array ▁ is " ) NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT print ( arr [ i ] , end = ' ▁ ' ) NEW_LINE DEDENT rotate ( arr , n ) NEW_LINE print ( " Rotated array is " ) NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT print ( arr [ i ] , end = ' ▁ ' ) NEW_LINE DEDENT
def rearrangeNaive ( arr , n ) : NEW_LINE
temp = [ 0 ] * n NEW_LINE
for i in range ( 0 , n ) : NEW_LINE INDENT temp [ arr [ i ] ] = i NEW_LINE DEDENT
for i in range ( 0 , n ) : NEW_LINE INDENT arr [ i ] = temp [ i ] NEW_LINE DEDENT
def printArray ( arr , n ) : NEW_LINE INDENT for i in range ( 0 , n ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT DEDENT
arr = [ 1 , 3 , 0 , 2 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Given ▁ array ▁ is " , end = " ▁ " ) NEW_LINE printArray ( arr , n ) NEW_LINE rearrangeNaive ( arr , n ) NEW_LINE print ( " Modified array is " , ▁ end ▁ = ▁ "   " ) NEW_LINE printArray ( arr , n ) NEW_LINE
def print2largest ( arr , arr_size ) : NEW_LINE
if ( arr_size < 2 ) : NEW_LINE INDENT print ( " ▁ Invalid ▁ Input ▁ " ) NEW_LINE return NEW_LINE DEDENT first = second = - 2147483648 NEW_LINE for i in range ( arr_size ) : NEW_LINE
if ( arr [ i ] > first ) : NEW_LINE INDENT second = first NEW_LINE first = arr [ i ] NEW_LINE DEDENT
elif ( arr [ i ] > second and arr [ i ] != first ) : NEW_LINE INDENT second = arr [ i ] NEW_LINE DEDENT if ( second == - 2147483648 ) : NEW_LINE print ( " There ▁ is ▁ no ▁ second ▁ largest ▁ element " ) NEW_LINE else : NEW_LINE print ( " The ▁ second ▁ largest ▁ element ▁ is " , second ) NEW_LINE
arr = [ 12 , 35 , 1 , 10 , 34 , 1 ] NEW_LINE n = len ( arr ) NEW_LINE print2largest ( arr , n ) NEW_LINE
class pair : NEW_LINE INDENT def __init__ ( self ) : NEW_LINE INDENT self . min = 0 NEW_LINE self . max = 0 NEW_LINE DEDENT DEDENT def getMinMax ( arr : list , n : int ) -> pair : NEW_LINE INDENT minmax = pair ( ) NEW_LINE DEDENT
if n == 1 : NEW_LINE INDENT minmax . max = arr [ 0 ] NEW_LINE minmax . min = arr [ 0 ] NEW_LINE return minmax NEW_LINE DEDENT
if arr [ 0 ] > arr [ 1 ] : NEW_LINE INDENT minmax . max = arr [ 0 ] NEW_LINE minmax . min = arr [ 1 ] NEW_LINE DEDENT else : NEW_LINE INDENT minmax . max = arr [ 1 ] NEW_LINE minmax . min = arr [ 0 ] NEW_LINE DEDENT for i in range ( 2 , n ) : NEW_LINE INDENT if arr [ i ] > minmax . max : NEW_LINE INDENT minmax . max = arr [ i ] NEW_LINE DEDENT elif arr [ i ] < minmax . min : NEW_LINE INDENT minmax . min = arr [ i ] NEW_LINE DEDENT DEDENT return minmax NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 1000 , 11 , 445 , 1 , 330 , 3000 ] NEW_LINE arr_size = 6 NEW_LINE minmax = getMinMax ( arr , arr_size ) NEW_LINE print ( " Minimum ▁ element ▁ is " , minmax . min ) NEW_LINE print ( " Maximum ▁ element ▁ is " , minmax . max ) NEW_LINE DEDENT
def getMinMax ( arr ) : NEW_LINE INDENT n = len ( arr ) NEW_LINE DEDENT
if ( n % 2 == 0 ) : NEW_LINE INDENT mx = max ( arr [ 0 ] , arr [ 1 ] ) NEW_LINE mn = min ( arr [ 0 ] , arr [ 1 ] ) NEW_LINE DEDENT
i = 2 NEW_LINE
else : NEW_LINE INDENT mx = mn = arr [ 0 ] NEW_LINE DEDENT
i = 1 NEW_LINE
while ( i < n - 1 ) : NEW_LINE INDENT if arr [ i ] < arr [ i + 1 ] : NEW_LINE INDENT mx = max ( mx , arr [ i + 1 ] ) NEW_LINE mn = min ( mn , arr [ i ] ) NEW_LINE DEDENT else : NEW_LINE INDENT mx = max ( mx , arr [ i ] ) NEW_LINE mn = min ( mn , arr [ i + 1 ] ) NEW_LINE DEDENT DEDENT
i += 2 NEW_LINE return ( mx , mn ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 1000 , 11 , 445 , 1 , 330 , 3000 ] NEW_LINE mx , mn = getMinMax ( arr ) NEW_LINE print ( " Minimum ▁ element ▁ is " , mn ) NEW_LINE print ( " Maximum ▁ element ▁ is " , mx ) NEW_LINE DEDENT
def minJumps ( arr , l , h ) : NEW_LINE
if ( h == l ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
if ( arr [ l ] == 0 ) : NEW_LINE INDENT return float ( ' inf ' ) NEW_LINE DEDENT
min = float ( ' inf ' ) NEW_LINE for i in range ( l + 1 , h + 1 ) : NEW_LINE INDENT if ( i < l + arr [ l ] + 1 ) : NEW_LINE INDENT jumps = minJumps ( arr , i , h ) NEW_LINE if ( jumps != float ( ' inf ' ) and jumps + 1 < min ) : NEW_LINE INDENT min = jumps + 1 NEW_LINE DEDENT DEDENT DEDENT return min NEW_LINE
arr = [ 1 , 3 , 6 , 3 , 2 , 3 , 6 , 8 , 9 , 5 ] NEW_LINE n = len ( arr ) NEW_LINE print ( ' Minimum ▁ number ▁ of ▁ jumps ▁ to ▁ reach ' , ' end ▁ is ' , minJumps ( arr , 0 , n - 1 ) ) NEW_LINE
def smallestSubWithSum ( arr , n , x ) : NEW_LINE
min_len = n + 1 NEW_LINE
for start in range ( 0 , n ) : NEW_LINE
curr_sum = arr [ start ] NEW_LINE
if ( curr_sum > x ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT
for end in range ( start + 1 , n ) : NEW_LINE
curr_sum += arr [ end ] NEW_LINE
if curr_sum > x and ( end - start + 1 ) < min_len : NEW_LINE INDENT min_len = ( end - start + 1 ) NEW_LINE DEDENT return min_len ; NEW_LINE
arr1 = [ 1 , 4 , 45 , 6 , 10 , 19 ] NEW_LINE x = 51 NEW_LINE n1 = len ( arr1 ) NEW_LINE res1 = smallestSubWithSum ( arr1 , n1 , x ) ; NEW_LINE if res1 == n1 + 1 : NEW_LINE INDENT print ( " Not ▁ possible " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( res1 ) NEW_LINE DEDENT arr2 = [ 1 , 10 , 5 , 2 , 7 ] NEW_LINE n2 = len ( arr2 ) NEW_LINE x = 9 NEW_LINE res2 = smallestSubWithSum ( arr2 , n2 , x ) ; NEW_LINE if res2 == n2 + 1 : NEW_LINE INDENT print ( " Not ▁ possible " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( res2 ) NEW_LINE DEDENT arr3 = [ 1 , 11 , 100 , 1 , 0 , 200 , 3 , 2 , 1 , 250 ] NEW_LINE n3 = len ( arr3 ) NEW_LINE x = 280 NEW_LINE res3 = smallestSubWithSum ( arr3 , n3 , x ) NEW_LINE if res3 == n3 + 1 : NEW_LINE INDENT print ( " Not ▁ possible " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( res3 ) NEW_LINE DEDENT
class Node : NEW_LINE INDENT def __init__ ( self , key ) : NEW_LINE INDENT self . left = None NEW_LINE self . right = None NEW_LINE self . val = key NEW_LINE DEDENT DEDENT
def printPostorder ( root ) : NEW_LINE INDENT if root : NEW_LINE DEDENT
printPostorder ( root . left ) NEW_LINE
printPostorder ( root . right ) NEW_LINE
print ( root . val ) , NEW_LINE
def printInorder ( root ) : NEW_LINE INDENT if root : NEW_LINE DEDENT
printInorder ( root . left ) NEW_LINE
print ( root . val ) , NEW_LINE
printInorder ( root . right ) NEW_LINE
def printPreorder ( root ) : NEW_LINE INDENT if root : NEW_LINE DEDENT
print ( root . val ) , NEW_LINE
printPreorder ( root . left ) NEW_LINE
printPreorder ( root . right ) NEW_LINE
root = Node ( 1 ) NEW_LINE root . left = Node ( 2 ) NEW_LINE root . right = Node ( 3 ) NEW_LINE root . left . left = Node ( 4 ) NEW_LINE root . left . right = Node ( 5 ) NEW_LINE print " Preorder ▁ traversal ▁ of ▁ binary ▁ tree ▁ is " NEW_LINE printPreorder ( root ) NEW_LINE print   " NEW_LINE Inorder traversal of binary tree is " NEW_LINE printInorder ( root ) NEW_LINE print   " NEW_LINE Postorder traversal of binary tree is " NEW_LINE printPostorder ( root ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def inorder ( root ) : NEW_LINE INDENT if root is None : NEW_LINE INDENT return NEW_LINE DEDENT inorder ( root . left ) NEW_LINE print ( root . data , " " , end = " " ) NEW_LINE inorder ( root . right ) NEW_LINE DEDENT
def prune ( root , sum ) : NEW_LINE
if root is None : NEW_LINE INDENT return None NEW_LINE DEDENT
root . left = prune ( root . left , sum - root . data ) NEW_LINE root . right = prune ( root . right , sum - root . data ) NEW_LINE
if root . left is None and root . right is None : NEW_LINE INDENT if sum > root . data : NEW_LINE INDENT return None NEW_LINE DEDENT DEDENT return root NEW_LINE
root = Node ( 1 ) NEW_LINE root . left = Node ( 2 ) NEW_LINE root . right = Node ( 3 ) NEW_LINE root . left . left = Node ( 4 ) NEW_LINE root . left . right = Node ( 5 ) NEW_LINE root . right . left = Node ( 6 ) NEW_LINE root . right . right = Node ( 7 ) NEW_LINE root . left . left . left = Node ( 8 ) NEW_LINE root . left . left . right = Node ( 9 ) NEW_LINE root . left . right . left = Node ( 12 ) NEW_LINE root . right . right . left = Node ( 10 ) NEW_LINE root . right . right . left . right = Node ( 11 ) NEW_LINE root . left . left . right . left = Node ( 13 ) NEW_LINE root . left . left . right . right = Node ( 14 ) NEW_LINE root . left . left . right . right . left = Node ( 15 ) NEW_LINE print ( " Tree ▁ before ▁ truncation " ) NEW_LINE inorder ( root ) NEW_LINE prune ( root , 45 ) NEW_LINE print ( " Tree after truncation " ) NEW_LINE inorder ( root ) NEW_LINE
NA = - 1 NEW_LINE
def moveToEnd ( mPlusN , size ) : NEW_LINE INDENT i = 0 NEW_LINE j = size - 1 NEW_LINE for i in range ( size - 1 , - 1 , - 1 ) : NEW_LINE INDENT if ( mPlusN [ i ] != NA ) : NEW_LINE INDENT mPlusN [ j ] = mPlusN [ i ] NEW_LINE j -= 1 NEW_LINE DEDENT DEDENT DEDENT
def merge ( mPlusN , N , m , n ) : NEW_LINE INDENT i = n NEW_LINE DEDENT
j = 0 NEW_LINE
k = 0 NEW_LINE
while ( k < ( m + n ) ) : NEW_LINE
if ( ( j == n ) or ( i < ( m + n ) and mPlusN [ i ] <= N [ j ] ) ) : NEW_LINE INDENT mPlusN [ k ] = mPlusN [ i ] NEW_LINE k += 1 NEW_LINE i += 1 NEW_LINE DEDENT
else : NEW_LINE INDENT mPlusN [ k ] = N [ j ] NEW_LINE k += 1 NEW_LINE j += 1 NEW_LINE DEDENT
def printArray ( arr , size ) : NEW_LINE INDENT for i in range ( size ) : NEW_LINE INDENT print ( arr [ i ] , " ▁ " , end = " " ) NEW_LINE DEDENT print ( ) NEW_LINE DEDENT
mPlusN = [ 2 , 8 , NA , NA , NA , 13 , NA , 15 , 20 ] NEW_LINE N = [ 5 , 7 , 9 , 25 ] NEW_LINE n = len ( N ) NEW_LINE m = len ( mPlusN ) - n NEW_LINE
moveToEnd ( mPlusN , m + n ) NEW_LINE
merge ( mPlusN , N , m , n ) NEW_LINE
printArray ( mPlusN , m + n ) NEW_LINE
def getCount ( n , K ) : NEW_LINE
if ( n == 1 ) : NEW_LINE INDENT return 10 NEW_LINE DEDENT
dp = [ 0 ] * 11 NEW_LINE
next = [ 0 ] * 11 NEW_LINE
for i in range ( 1 , 9 + 1 ) : NEW_LINE INDENT dp [ i ] = 1 NEW_LINE DEDENT
for i in range ( 2 , n + 1 ) : NEW_LINE INDENT for j in range ( 9 + 1 ) : NEW_LINE DEDENT
l = max ( 0 , j - k ) NEW_LINE r = min ( 9 , j + k ) NEW_LINE
next [ l ] += dp [ j ] NEW_LINE next [ r + 1 ] -= dp [ j ] NEW_LINE
for j in range ( 1 , 9 + 1 ) : NEW_LINE INDENT next [ j ] += next [ j - 1 ] NEW_LINE DEDENT
for j in range ( 10 ) : NEW_LINE INDENT dp [ j ] = next [ j ] NEW_LINE next [ j ] = 0 NEW_LINE DEDENT
count = 0 NEW_LINE for i in range ( 9 + 1 ) : NEW_LINE INDENT count += dp [ i ] NEW_LINE DEDENT
return count NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 2 NEW_LINE k = 1 NEW_LINE print ( getCount ( n , k ) ) NEW_LINE DEDENT
def getInvCount ( arr , n ) : NEW_LINE INDENT inv_count = 0 NEW_LINE for i in range ( n ) : NEW_LINE INDENT for j in range ( i + 1 , n ) : NEW_LINE INDENT if ( arr [ i ] > arr [ j ] ) : NEW_LINE INDENT inv_count += 1 NEW_LINE DEDENT DEDENT DEDENT return inv_count NEW_LINE DEDENT
arr = [ 1 , 20 , 6 , 4 , 5 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Number ▁ of ▁ inversions ▁ are " , getInvCount ( arr , n ) ) NEW_LINE
def minAbsSumPair ( arr , arr_size ) : NEW_LINE INDENT inv_count = 0 NEW_LINE DEDENT
if arr_size < 2 : NEW_LINE INDENT print ( " Invalid ▁ Input " ) NEW_LINE return NEW_LINE DEDENT
min_l = 0 NEW_LINE min_r = 1 NEW_LINE min_sum = arr [ 0 ] + arr [ 1 ] NEW_LINE for l in range ( 0 , arr_size - 1 ) : NEW_LINE INDENT for r in range ( l + 1 , arr_size ) : NEW_LINE INDENT sum = arr [ l ] + arr [ r ] NEW_LINE if abs ( min_sum ) > abs ( sum ) : NEW_LINE INDENT min_sum = sum NEW_LINE min_l = l NEW_LINE min_r = r NEW_LINE DEDENT DEDENT DEDENT print ( " The ▁ two ▁ elements ▁ whose ▁ sum ▁ is ▁ minimum ▁ are " , arr [ min_l ] , " and ▁ " , arr [ min_r ] ) NEW_LINE
arr = [ 1 , 60 , - 10 , 70 , - 80 , 85 ] NEW_LINE minAbsSumPair ( arr , 6 ) ; NEW_LINE
def printUnion ( arr1 , arr2 , m , n ) : NEW_LINE INDENT i , j = 0 , 0 NEW_LINE while i < m and j < n : NEW_LINE INDENT if arr1 [ i ] < arr2 [ j ] : NEW_LINE INDENT print ( arr1 [ i ] ) NEW_LINE i += 1 NEW_LINE DEDENT elif arr2 [ j ] < arr1 [ i ] : NEW_LINE INDENT print ( arr2 [ j ] ) NEW_LINE j += 1 NEW_LINE DEDENT else : NEW_LINE INDENT print ( arr2 [ j ] ) NEW_LINE j += 1 NEW_LINE i += 1 NEW_LINE DEDENT DEDENT DEDENT
while i < m : NEW_LINE INDENT print ( arr1 [ i ] ) NEW_LINE i += 1 NEW_LINE DEDENT while j < n : NEW_LINE INDENT print ( arr2 [ j ] ) NEW_LINE j += 1 NEW_LINE DEDENT
arr1 = [ 1 , 2 , 4 , 5 , 6 ] NEW_LINE arr2 = [ 2 , 3 , 5 , 7 ] NEW_LINE m = len ( arr1 ) NEW_LINE n = len ( arr2 ) NEW_LINE printUnion ( arr1 , arr2 , m , n ) NEW_LINE
def printIntersection ( arr1 , arr2 , m , n ) : NEW_LINE INDENT i , j = 0 , 0 NEW_LINE while i < m and j < n : NEW_LINE INDENT if arr1 [ i ] < arr2 [ j ] : NEW_LINE INDENT i += 1 NEW_LINE DEDENT elif arr2 [ j ] < arr1 [ i ] : NEW_LINE INDENT j += 1 NEW_LINE DEDENT else : NEW_LINE INDENT print ( arr2 [ j ] ) NEW_LINE j += 1 NEW_LINE i += 1 NEW_LINE DEDENT DEDENT DEDENT
arr1 = [ 1 , 2 , 4 , 5 , 6 ] NEW_LINE arr2 = [ 2 , 3 , 5 , 7 ] NEW_LINE m = len ( arr1 ) NEW_LINE n = len ( arr2 ) NEW_LINE
printIntersection ( arr1 , arr2 , m , n ) NEW_LINE
class node : NEW_LINE INDENT def __init__ ( self ) : NEW_LINE INDENT self . data = 0 NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def printPath ( root , target_leaf ) : NEW_LINE
if ( root == None ) : NEW_LINE INDENT return False NEW_LINE DEDENT
if ( root == target_leaf or printPath ( root . left , target_leaf ) or printPath ( root . right , target_leaf ) ) : NEW_LINE INDENT print ( root . data , end = " ▁ " ) NEW_LINE return True NEW_LINE DEDENT return False NEW_LINE max_sum_ref = 0 NEW_LINE target_leaf_ref = None NEW_LINE
def getTargetLeaf ( Node , curr_sum ) : NEW_LINE INDENT global max_sum_ref NEW_LINE global target_leaf_ref NEW_LINE if ( Node == None ) : NEW_LINE INDENT return NEW_LINE DEDENT DEDENT
curr_sum = curr_sum + Node . data NEW_LINE
if ( Node . left == None and Node . right == None ) : NEW_LINE INDENT if ( curr_sum > max_sum_ref ) : NEW_LINE INDENT max_sum_ref = curr_sum NEW_LINE target_leaf_ref = Node NEW_LINE DEDENT DEDENT
getTargetLeaf ( Node . left , curr_sum ) NEW_LINE getTargetLeaf ( Node . right , curr_sum ) NEW_LINE
def maxSumPath ( Node ) : NEW_LINE INDENT global max_sum_ref NEW_LINE global target_leaf_ref NEW_LINE DEDENT
if ( Node == None ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT target_leaf_ref = None NEW_LINE max_sum_ref = - 32676 NEW_LINE
getTargetLeaf ( Node , 0 ) NEW_LINE
printPath ( Node , target_leaf_ref ) ; NEW_LINE
return max_sum_ref ; NEW_LINE
def newNode ( data ) : NEW_LINE INDENT temp = node ( ) ; NEW_LINE temp . data = data ; NEW_LINE temp . left = None ; NEW_LINE temp . right = None ; NEW_LINE return temp ; NEW_LINE DEDENT
root = None ; NEW_LINE root = newNode ( 10 ) ; NEW_LINE root . left = newNode ( - 2 ) ; NEW_LINE root . right = newNode ( 7 ) ; NEW_LINE root . left . left = newNode ( 8 ) ; NEW_LINE root . left . right = newNode ( - 4 ) ; NEW_LINE print ( " Following ▁ are ▁ the ▁ nodes ▁ on ▁ the ▁ maximum ▁ sum ▁ path ▁ " ) ; NEW_LINE sum = maxSumPath ( root ) ; NEW_LINE print (   " Sum of the nodes is   " , sum ) ; NEW_LINE
def sort012 ( a , arr_size ) : NEW_LINE INDENT lo = 0 NEW_LINE hi = arr_size - 1 NEW_LINE mid = 0 NEW_LINE while mid <= hi : NEW_LINE INDENT if a [ mid ] == 0 : NEW_LINE INDENT a [ lo ] , a [ mid ] = a [ mid ] , a [ lo ] NEW_LINE lo = lo + 1 NEW_LINE mid = mid + 1 NEW_LINE DEDENT elif a [ mid ] == 1 : NEW_LINE INDENT mid = mid + 1 NEW_LINE DEDENT else : NEW_LINE INDENT a [ mid ] , a [ hi ] = a [ hi ] , a [ mid ] NEW_LINE hi = hi - 1 NEW_LINE DEDENT DEDENT return a NEW_LINE DEDENT
def printArray ( a ) : NEW_LINE INDENT for k in a : NEW_LINE INDENT print k , NEW_LINE DEDENT DEDENT
arr = [ 0 , 1 , 1 , 0 , 1 , 2 , 1 , 2 , 0 , 0 , 0 , 1 ] NEW_LINE arr_size = len ( arr ) NEW_LINE arr = sort012 ( arr , arr_size ) NEW_LINE print   " Array after segregation : NEW_LINE " , NEW_LINE printArray ( arr ) NEW_LINE
def printUnsorted ( arr , n ) : NEW_LINE INDENT e = n - 1 NEW_LINE DEDENT
for s in range ( 0 , n - 1 ) : NEW_LINE INDENT if arr [ s ] > arr [ s + 1 ] : NEW_LINE INDENT break NEW_LINE DEDENT DEDENT if s == n - 1 : NEW_LINE INDENT print ( " The ▁ complete ▁ array ▁ is ▁ sorted " ) NEW_LINE exit ( ) NEW_LINE DEDENT
e = n - 1 NEW_LINE while e > 0 : NEW_LINE INDENT if arr [ e ] < arr [ e - 1 ] : NEW_LINE INDENT break NEW_LINE DEDENT e -= 1 NEW_LINE DEDENT
max = arr [ s ] NEW_LINE min = arr [ s ] NEW_LINE for i in range ( s + 1 , e + 1 ) : NEW_LINE INDENT if arr [ i ] > max : NEW_LINE INDENT max = arr [ i ] NEW_LINE DEDENT if arr [ i ] < min : NEW_LINE INDENT min = arr [ i ] NEW_LINE DEDENT DEDENT
for i in range ( s ) : NEW_LINE INDENT if arr [ i ] > min : NEW_LINE INDENT s = i NEW_LINE break NEW_LINE DEDENT DEDENT
i = n - 1 NEW_LINE while i >= e + 1 : NEW_LINE INDENT if arr [ i ] < max : NEW_LINE INDENT e = i NEW_LINE break NEW_LINE DEDENT i -= 1 NEW_LINE DEDENT
print ( " The ▁ unsorted ▁ subarray ▁ which ▁ makes ▁ the ▁ given ▁ array " ) NEW_LINE print ( " sorted ▁ lies ▁ between ▁ the ▁ indexes ▁ % d ▁ and ▁ % d " % ( s , e ) ) NEW_LINE arr = [ 10 , 12 , 20 , 30 , 25 , 40 , 32 , 31 , 35 , 50 , 60 ] NEW_LINE arr_size = len ( arr ) NEW_LINE printUnsorted ( arr , arr_size ) NEW_LINE
def findnumberofTriangles ( arr ) : NEW_LINE INDENT n = len ( arr ) NEW_LINE DEDENT
arr . sort ( ) NEW_LINE
count = 0 NEW_LINE
for i in range ( 0 , n - 2 ) : NEW_LINE
k = i + 2 NEW_LINE
for j in range ( i + 1 , n ) : NEW_LINE
while ( k < n and arr [ i ] + arr [ j ] > arr [ k ] ) : NEW_LINE INDENT k += 1 NEW_LINE DEDENT
if ( k > j ) : NEW_LINE INDENT count += k - j - 1 NEW_LINE DEDENT return count NEW_LINE
arr = [ 10 , 21 , 22 , 100 , 101 , 200 , 300 ] NEW_LINE print " Number ▁ of ▁ Triangles : " , findnumberofTriangles ( arr ) NEW_LINE
def findElement ( arr , n , key ) : NEW_LINE INDENT for i in range ( n ) : NEW_LINE INDENT if ( arr [ i ] == key ) : NEW_LINE INDENT return i NEW_LINE DEDENT DEDENT return - 1 NEW_LINE DEDENT
arr = [ 12 , 34 , 10 , 6 , 40 ] NEW_LINE n = len ( arr ) NEW_LINE
key = 40 NEW_LINE index = findElement ( arr , n , key ) NEW_LINE if index != - 1 : NEW_LINE INDENT print ( " element ▁ found ▁ at ▁ position : ▁ " + str ( index + 1 ) ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " element ▁ not ▁ found " ) NEW_LINE DEDENT
def binarySearch ( arr , low , high , key ) : NEW_LINE
mid = ( low + high ) / 2 NEW_LINE if ( key == arr [ int ( mid ) ] ) : NEW_LINE INDENT return mid NEW_LINE DEDENT if ( key > arr [ int ( mid ) ] ) : NEW_LINE INDENT return binarySearch ( arr , ( mid + 1 ) , high , key ) NEW_LINE DEDENT if ( key < arr [ int ( mid ) ] ) : NEW_LINE INDENT return binarySearch ( arr , low , ( mid - 1 ) , key ) NEW_LINE DEDENT return 0 NEW_LINE
arr = [ 5 , 6 , 7 , 8 , 9 , 10 ] NEW_LINE n = len ( arr ) NEW_LINE key = 10 NEW_LINE print ( " Index : " , int ( binarySearch ( arr , 0 , n - 1 , key ) ) ) NEW_LINE
def insertSorted ( arr , n , key , capacity ) : NEW_LINE
if ( n >= capacity ) : NEW_LINE INDENT return n NEW_LINE DEDENT i = n - 1 NEW_LINE while i >= 0 and arr [ i ] > key : NEW_LINE INDENT arr [ i + 1 ] = arr [ i ] NEW_LINE i -= 1 NEW_LINE DEDENT arr [ i + 1 ] = key NEW_LINE return ( n + 1 ) NEW_LINE
arr = [ 12 , 16 , 20 , 40 , 50 , 70 ] NEW_LINE for i in range ( 20 ) : NEW_LINE INDENT arr . append ( 0 ) NEW_LINE DEDENT capacity = len ( arr ) NEW_LINE n = 6 NEW_LINE key = 26 NEW_LINE print ( " Before ▁ Insertion : ▁ " , end = " ▁ " ) ; NEW_LINE for i in range ( n ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT
n = insertSorted ( arr , n , key , capacity ) NEW_LINE print ( " After Insertion : " , ▁ end ▁ = ▁ " " ) NEW_LINE for i in range ( n ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT
def binarySearch ( arr , low , high , key ) : NEW_LINE INDENT if ( high < low ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT mid = ( low + high ) // 2 NEW_LINE if ( key == arr [ mid ] ) : NEW_LINE INDENT return mid NEW_LINE DEDENT if ( key > arr [ mid ] ) : NEW_LINE INDENT return binarySearch ( arr , ( mid + 1 ) , high , key ) NEW_LINE DEDENT return binarySearch ( arr , low , ( mid - 1 ) , key ) NEW_LINE DEDENT
def deleteElement ( arr , n , key ) : NEW_LINE
pos = binarySearch ( arr , 0 , n - 1 , key ) NEW_LINE if ( pos == - 1 ) : NEW_LINE INDENT print ( " Element ▁ not ▁ found " ) NEW_LINE return n NEW_LINE DEDENT
for i in range ( pos , n - 1 ) : NEW_LINE INDENT arr [ i ] = arr [ i + 1 ] NEW_LINE DEDENT return n - 1 NEW_LINE
arr = [ 10 , 20 , 30 , 40 , 50 ] NEW_LINE n = len ( arr ) NEW_LINE key = 30 NEW_LINE print ( " Array ▁ before ▁ deletion " ) NEW_LINE for i in range ( n ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT n = deleteElement ( arr , n , key ) NEW_LINE print ( " Array after deletion " ) NEW_LINE for i in range ( n ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT
def equilibrium ( arr ) : NEW_LINE INDENT leftsum = 0 NEW_LINE rightsum = 0 NEW_LINE n = len ( arr ) NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT leftsum = 0 NEW_LINE rightsum = 0 NEW_LINE DEDENT
for j in range ( i ) : NEW_LINE INDENT leftsum += arr [ j ] NEW_LINE DEDENT
for j in range ( i + 1 , n ) : NEW_LINE INDENT rightsum += arr [ j ] NEW_LINE DEDENT
if leftsum == rightsum : NEW_LINE INDENT return i NEW_LINE DEDENT
return - 1 NEW_LINE
arr = [ - 7 , 1 , 5 , 2 , - 4 , 3 , 0 ] NEW_LINE print ( equilibrium ( arr ) ) NEW_LINE
def equilibrium ( arr ) : NEW_LINE
total_sum = sum ( arr ) NEW_LINE leftsum = 0 NEW_LINE for i , num in enumerate ( arr ) : NEW_LINE
total_sum -= num NEW_LINE if leftsum == total_sum : NEW_LINE INDENT return i NEW_LINE DEDENT leftsum += num NEW_LINE
return - 1 NEW_LINE
arr = [ - 7 , 1 , 5 , 2 , - 4 , 3 , 0 ] NEW_LINE print ( ' First ▁ equilibrium ▁ index ▁ is ▁ ' , equilibrium ( arr ) ) NEW_LINE
def ceilSearch ( arr , low , high , x ) : NEW_LINE
if x <= arr [ low ] : NEW_LINE INDENT return low NEW_LINE DEDENT
i = low NEW_LINE for i in range ( high ) : NEW_LINE INDENT if arr [ i ] == x : NEW_LINE INDENT return i NEW_LINE DEDENT DEDENT
if arr [ i ] < x and arr [ i + 1 ] >= x : NEW_LINE INDENT return i + 1 NEW_LINE DEDENT
return - 1 NEW_LINE
arr = [ 1 , 2 , 8 , 10 , 10 , 12 , 19 ] NEW_LINE n = len ( arr ) NEW_LINE x = 3 NEW_LINE index = ceilSearch ( arr , 0 , n - 1 , x ) ; NEW_LINE if index == - 1 : NEW_LINE INDENT print ( " Ceiling ▁ of ▁ % d ▁ doesn ' t ▁ exist ▁ in ▁ array ▁ " % x ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " ceiling ▁ of ▁ % d ▁ is ▁ % d " % ( x , arr [ index ] ) ) NEW_LINE DEDENT
def ceilSearch ( arr , low , high , x ) : NEW_LINE
if x <= arr [ low ] : NEW_LINE INDENT return low NEW_LINE DEDENT
if x > arr [ high ] : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
mid = ( low + high ) / 2 ; NEW_LINE
if arr [ mid ] == x : NEW_LINE INDENT return mid NEW_LINE DEDENT
elif arr [ mid ] < x : NEW_LINE INDENT if mid + 1 <= high and x <= arr [ mid + 1 ] : NEW_LINE INDENT return mid + 1 NEW_LINE DEDENT else : NEW_LINE INDENT return ceilSearch ( arr , mid + 1 , high , x ) NEW_LINE DEDENT DEDENT
else : NEW_LINE INDENT if mid - 1 >= low and x > arr [ mid - 1 ] : NEW_LINE INDENT return mid NEW_LINE DEDENT else : NEW_LINE INDENT return ceilSearch ( arr , low , mid - 1 , x ) NEW_LINE DEDENT DEDENT
arr = [ 1 , 2 , 8 , 10 , 10 , 12 , 19 ] NEW_LINE n = len ( arr ) NEW_LINE x = 20 NEW_LINE index = ceilSearch ( arr , 0 , n - 1 , x ) ; NEW_LINE if index == - 1 : NEW_LINE INDENT print ( " Ceiling ▁ of ▁ % d ▁ doesn ' t ▁ exist ▁ in ▁ array ▁ " % x ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " ceiling ▁ of ▁ % d ▁ is ▁ % d " % ( x , arr [ index ] ) ) NEW_LINE DEDENT
def isPairSum ( A , N , X ) : NEW_LINE INDENT for i in range ( N ) : NEW_LINE INDENT for j in range ( N ) : NEW_LINE DEDENT DEDENT
if ( i == j ) : NEW_LINE INDENT continue NEW_LINE DEDENT
if ( A [ i ] + A [ j ] == X ) : NEW_LINE INDENT return True NEW_LINE DEDENT
if ( A [ i ] + A [ j ] > X ) : NEW_LINE INDENT break NEW_LINE DEDENT
return 0 NEW_LINE
arr = [ 3 , 5 , 9 , 2 , 8 , 10 , 11 ] NEW_LINE val = 17 NEW_LINE
print ( isPairSum ( arr , len ( arr ) , val ) ) NEW_LINE
def isPairSum ( A , N , X ) : NEW_LINE
i = 0 NEW_LINE
j = N - 1 NEW_LINE while ( i < j ) : NEW_LINE
if ( A [ i ] + A [ j ] == X ) : NEW_LINE INDENT return True NEW_LINE DEDENT
elif ( A [ i ] + A [ j ] < X ) : NEW_LINE INDENT i += 1 NEW_LINE DEDENT
else : NEW_LINE INDENT j -= 1 NEW_LINE DEDENT return 0 NEW_LINE
arr = [ 3 , 5 , 9 , 2 , 8 , 10 , 11 ] NEW_LINE
val = 17 NEW_LINE
print ( isPairSum ( arr , len ( arr ) , val ) ) NEW_LINE
def carAssembly ( a , t , e , x ) : NEW_LINE INDENT NUM_STATION = len ( a [ 0 ] ) NEW_LINE T1 = [ 0 for i in range ( NUM_STATION ) ] NEW_LINE T2 = [ 0 for i in range ( NUM_STATION ) ] NEW_LINE DEDENT
T1 [ 0 ] = e [ 0 ] + a [ 0 ] [ 0 ] NEW_LINE
T2 [ 0 ] = e [ 1 ] + a [ 1 ] [ 0 ] NEW_LINE
for i in range ( 1 , NUM_STATION ) : NEW_LINE INDENT T1 [ i ] = min ( T1 [ i - 1 ] + a [ 0 ] [ i ] , T2 [ i - 1 ] + t [ 1 ] [ i ] + a [ 0 ] [ i ] ) NEW_LINE T2 [ i ] = min ( T2 [ i - 1 ] + a [ 1 ] [ i ] , T1 [ i - 1 ] + t [ 0 ] [ i ] + a [ 1 ] [ i ] ) NEW_LINE DEDENT
return min ( T1 [ NUM_STATION - 1 ] + x [ 0 ] , T2 [ NUM_STATION - 1 ] + x [ 1 ] ) NEW_LINE
a = [ [ 4 , 5 , 3 , 2 ] , [ 2 , 10 , 1 , 4 ] ] NEW_LINE t = [ [ 0 , 7 , 4 , 5 ] , [ 0 , 9 , 2 , 8 ] ] NEW_LINE e = [ 10 , 12 ] NEW_LINE x = [ 18 , 7 ] NEW_LINE print ( carAssembly ( a , t , e , x ) ) NEW_LINE
def Min ( a , b ) : NEW_LINE INDENT return min ( a , b ) NEW_LINE DEDENT
def findMinInsertionsDP ( str1 , n ) : NEW_LINE
table = [ [ 0 for i in range ( n ) ] for i in range ( n ) ] NEW_LINE l , h , gap = 0 , 0 , 0 NEW_LINE
for gap in range ( 1 , n ) : NEW_LINE INDENT l = 0 NEW_LINE for h in range ( gap , n ) : NEW_LINE INDENT if str1 [ l ] == str1 [ h ] : NEW_LINE INDENT table [ l ] [ h ] = table [ l + 1 ] [ h - 1 ] NEW_LINE DEDENT else : NEW_LINE INDENT table [ l ] [ h ] = ( Min ( table [ l ] [ h - 1 ] , table [ l + 1 ] [ h ] ) + 1 ) NEW_LINE DEDENT l += 1 NEW_LINE DEDENT DEDENT
return table [ 0 ] [ n - 1 ] ; NEW_LINE
str1 = " geeks " NEW_LINE print ( findMinInsertionsDP ( str1 , len ( str1 ) ) ) NEW_LINE
def max ( x , y ) : NEW_LINE INDENT if ( x > y ) : NEW_LINE INDENT return x NEW_LINE DEDENT else : NEW_LINE INDENT return y NEW_LINE DEDENT DEDENT
class node : NEW_LINE INDENT def __init__ ( self ) : NEW_LINE INDENT self . data = 0 NEW_LINE self . left = self . right = None NEW_LINE DEDENT DEDENT
def LISS ( root ) : NEW_LINE INDENT if ( root == None ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT DEDENT
size_excl = LISS ( root . left ) + LISS ( root . right ) NEW_LINE
size_incl = 1 NEW_LINE if ( root . left != None ) : NEW_LINE INDENT size_incl += LISS ( root . left . left ) + LISS ( root . left . right ) NEW_LINE DEDENT if ( root . right != None ) : NEW_LINE INDENT size_incl += LISS ( root . right . left ) + LISS ( root . right . right ) NEW_LINE DEDENT
return max ( size_incl , size_excl ) NEW_LINE
def newNode ( data ) : NEW_LINE INDENT temp = node ( ) NEW_LINE temp . data = data NEW_LINE temp . left = temp . right = None NEW_LINE return temp NEW_LINE DEDENT
root = newNode ( 20 ) NEW_LINE root . left = newNode ( 8 ) NEW_LINE root . left . left = newNode ( 4 ) NEW_LINE root . left . right = newNode ( 12 ) NEW_LINE root . left . right . left = newNode ( 10 ) NEW_LINE root . left . right . right = newNode ( 14 ) NEW_LINE root . right = newNode ( 22 ) NEW_LINE root . right . right = newNode ( 25 ) NEW_LINE print ( " Size ▁ of ▁ the ▁ Largest " , " ▁ Independent ▁ Set ▁ is ▁ " , LISS ( root ) ) NEW_LINE
class Pair ( object ) : NEW_LINE INDENT def __init__ ( self , a , b ) : NEW_LINE INDENT self . a = a NEW_LINE self . b = b NEW_LINE DEDENT DEDENT
def maxChainLength ( arr , n ) : NEW_LINE INDENT max = 0 NEW_LINE DEDENT
mcl = [ 1 for i in range ( n ) ] NEW_LINE
for i in range ( 1 , n ) : NEW_LINE INDENT for j in range ( 0 , i ) : NEW_LINE INDENT if ( arr [ i ] . a > arr [ j ] . b and mcl [ i ] < mcl [ j ] + 1 ) : NEW_LINE INDENT mcl [ i ] = mcl [ j ] + 1 NEW_LINE DEDENT DEDENT DEDENT
for i in range ( n ) : NEW_LINE INDENT if ( max < mcl [ i ] ) : NEW_LINE INDENT max = mcl [ i ] NEW_LINE DEDENT DEDENT return max NEW_LINE
arr = [ Pair ( 5 , 24 ) , Pair ( 15 , 25 ) , Pair ( 27 , 40 ) , Pair ( 50 , 60 ) ] NEW_LINE print ( ' Length ▁ of ▁ maximum ▁ size ▁ chain ▁ is ' , maxChainLength ( arr , len ( arr ) ) ) NEW_LINE
def minPalPartion ( str ) : NEW_LINE
n = len ( str ) NEW_LINE
C = [ [ 0 for i in range ( n ) ] for i in range ( n ) ] NEW_LINE P = [ [ False for i in range ( n ) ] for i in range ( n ) ] NEW_LINE
j = 0 NEW_LINE k = 0 NEW_LINE L = 0 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT P [ i ] [ i ] = True ; NEW_LINE C [ i ] [ i ] = 0 ; NEW_LINE DEDENT
for L in range ( 2 , n + 1 ) : NEW_LINE
for i in range ( n - L + 1 ) : NEW_LINE
j = i + L - 1 NEW_LINE
if L == 2 : NEW_LINE INDENT P [ i ] [ j ] = ( str [ i ] == str [ j ] ) NEW_LINE DEDENT else : NEW_LINE INDENT P [ i ] [ j ] = ( ( str [ i ] == str [ j ] ) and P [ i + 1 ] [ j - 1 ] ) NEW_LINE DEDENT
if P [ i ] [ j ] == True : NEW_LINE INDENT C [ i ] [ j ] = 0 NEW_LINE DEDENT else : NEW_LINE
C [ i ] [ j ] = 100000000 NEW_LINE for k in range ( i , j ) : NEW_LINE INDENT C [ i ] [ j ] = min ( C [ i ] [ j ] , C [ i ] [ k ] + C [ k + 1 ] [ j ] + 1 ) NEW_LINE DEDENT
return C [ 0 ] [ n - 1 ] NEW_LINE
str = " ababbbabbababa " NEW_LINE print ( ' Min ▁ cuts ▁ needed ▁ for ▁ Palindrome ▁ Partitioning ▁ is ' , minPalPartion ( str ) ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def getLevelDiff ( root ) : NEW_LINE
if root is None : NEW_LINE INDENT return 0 NEW_LINE DEDENT
return ( root . data - getLevelDiff ( root . left ) - getLevelDiff ( root . right ) ) NEW_LINE
root = Node ( 5 ) NEW_LINE root . left = Node ( 2 ) NEW_LINE root . right = Node ( 6 ) NEW_LINE root . left . left = Node ( 1 ) NEW_LINE root . left . right = Node ( 4 ) NEW_LINE root . left . right . left = Node ( 3 ) NEW_LINE root . right . right = Node ( 8 ) NEW_LINE root . right . right . right = Node ( 9 ) NEW_LINE root . right . right . left = Node ( 7 ) NEW_LINE print " % d ▁ is ▁ the ▁ required ▁ difference " % ( getLevelDiff ( root ) ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def hasPathSum ( node , s ) : NEW_LINE INDENT if node is None : NEW_LINE INDENT return ( s == 0 ) NEW_LINE DEDENT else : NEW_LINE INDENT ans = 0 NEW_LINE subSum = s - node . data NEW_LINE if ( subSum == 0 and node . left == None and node . right == None ) : NEW_LINE INDENT return True NEW_LINE DEDENT if node . left is not None : NEW_LINE INDENT ans = ans or hasPathSum ( node . left , subSum ) NEW_LINE DEDENT if node . right is not None : NEW_LINE INDENT ans = ans or hasPathSum ( node . right , subSum ) NEW_LINE DEDENT return ans NEW_LINE DEDENT DEDENT
s = 21 NEW_LINE root = Node ( 10 ) NEW_LINE root . left = Node ( 8 ) NEW_LINE root . right = Node ( 2 ) NEW_LINE root . left . right = Node ( 5 ) NEW_LINE root . left . left = Node ( 3 ) NEW_LINE root . right . left = Node ( 2 ) NEW_LINE if hasPathSum ( root , s ) : NEW_LINE INDENT print " There ▁ is ▁ a ▁ root - to - leaf ▁ path ▁ with ▁ sum ▁ % d " % ( s ) NEW_LINE DEDENT else : NEW_LINE INDENT print " There ▁ is ▁ no ▁ root - to - leaf ▁ path ▁ with ▁ sum ▁ % d " % ( s ) NEW_LINE DEDENT
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def treePathsSumUtil ( root , val ) : NEW_LINE
if root is None : NEW_LINE INDENT return 0 NEW_LINE DEDENT
val = ( val * 10 + root . data ) NEW_LINE
if root . left is None and root . right is None : NEW_LINE INDENT return val NEW_LINE DEDENT
return ( treePathsSumUtil ( root . left , val ) + treePathsSumUtil ( root . right , val ) ) NEW_LINE
def treePathsSum ( root ) : NEW_LINE
return treePathsSumUtil ( root , 0 ) NEW_LINE
root = Node ( 6 ) NEW_LINE root . left = Node ( 3 ) NEW_LINE root . right = Node ( 5 ) NEW_LINE root . left . left = Node ( 2 ) NEW_LINE root . left . right = Node ( 5 ) NEW_LINE root . right . right = Node ( 4 ) NEW_LINE root . left . right . left = Node ( 7 ) NEW_LINE root . left . right . right = Node ( 4 ) NEW_LINE print " Sum ▁ of ▁ all ▁ paths ▁ is " , treePathsSum ( root ) NEW_LINE
def minChocolates ( a , n ) : NEW_LINE INDENT i , j = 0 , 0 NEW_LINE val , res = 1 , 0 NEW_LINE while ( j < n - 1 ) : NEW_LINE INDENT if ( a [ j ] > a [ j + 1 ] ) : NEW_LINE DEDENT DEDENT
j += 1 NEW_LINE continue NEW_LINE if ( i == j ) : NEW_LINE
res += val NEW_LINE else : NEW_LINE
res += get_sum ( val , i , j ) NEW_LINE
if ( a [ j ] < a [ j + 1 ] ) : NEW_LINE
val += 1 NEW_LINE else : NEW_LINE
val = 1 NEW_LINE j += 1 NEW_LINE i = j NEW_LINE
if ( i == j ) : NEW_LINE INDENT res += val NEW_LINE DEDENT else : NEW_LINE INDENT res += get_sum ( val , i , j ) NEW_LINE DEDENT return res NEW_LINE
def get_sum ( peak , start , end ) : NEW_LINE
count = end - start + 1 NEW_LINE
peak = max ( peak , count ) NEW_LINE
s = peak + ( ( ( count - 1 ) * count ) >> 1 ) NEW_LINE return s NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT a = [ 5 , 5 , 4 , 3 , 2 , 1 ] NEW_LINE n = len ( a ) NEW_LINE print ( ' Minimum ▁ number ▁ of ▁ chocolates ▁ = ' , minChocolates ( a , n ) ) NEW_LINE DEDENT
def sum ( n ) : NEW_LINE INDENT i = 1 NEW_LINE s = 0.0 NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE INDENT s = s + 1 / i ; NEW_LINE DEDENT return s ; NEW_LINE DEDENT
n = 5 NEW_LINE print ( " Sum ▁ is " , round ( sum ( n ) , 6 ) ) NEW_LINE
from math import pow NEW_LINE
def nthTermOfTheSeries ( n ) : NEW_LINE
if ( n % 2 == 0 ) : NEW_LINE INDENT nthTerm = pow ( n - 1 , 2 ) + n NEW_LINE DEDENT
else : NEW_LINE INDENT nthTerm = pow ( n + 1 , 2 ) + n NEW_LINE DEDENT
return nthTerm NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 8 NEW_LINE print ( int ( nthTermOfTheSeries ( n ) ) ) NEW_LINE n = 12 NEW_LINE print ( int ( nthTermOfTheSeries ( n ) ) ) NEW_LINE n = 102 NEW_LINE print ( int ( nthTermOfTheSeries ( n ) ) ) NEW_LINE n = 999 NEW_LINE print ( int ( nthTermOfTheSeries ( n ) ) ) NEW_LINE n = 9999 NEW_LINE print ( int ( nthTermOfTheSeries ( n ) ) ) NEW_LINE DEDENT
def Log2n ( n ) : NEW_LINE INDENT return 1 + Log2n ( n / 2 ) if ( n > 1 ) else 0 NEW_LINE DEDENT
n = 32 NEW_LINE print ( Log2n ( n ) ) NEW_LINE
def findAmount ( X , W , Y ) : NEW_LINE INDENT return ( X * ( Y - W ) / ( 100 - Y ) ) NEW_LINE DEDENT
X = 100 NEW_LINE W = 50 ; Y = 60 NEW_LINE print ( " Water ▁ to ▁ be ▁ added " , findAmount ( X , W , Y ) ) NEW_LINE
def AvgofSquareN ( n ) : NEW_LINE INDENT return ( ( n + 1 ) * ( 2 * n + 1 ) ) / 6 ; NEW_LINE DEDENT
n = 2 ; NEW_LINE print ( AvgofSquareN ( n ) ) ; NEW_LINE
def triangular_series ( n ) : NEW_LINE INDENT for i in range ( 1 , n + 1 ) : NEW_LINE INDENT print ( i * ( i + 1 ) // 2 , end = ' ▁ ' ) NEW_LINE DEDENT DEDENT
n = 5 NEW_LINE triangular_series ( n ) NEW_LINE
' NEW_LINE
' NEW_LINE def divisorSum ( n ) : NEW_LINE INDENT sum = 0 NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE INDENT sum += int ( n / i ) * i NEW_LINE DEDENT return int ( sum ) NEW_LINE DEDENT
n = 4 NEW_LINE print ( divisorSum ( n ) ) NEW_LINE n = 5 NEW_LINE print ( divisorSum ( n ) ) NEW_LINE
def SUM ( x , n ) : NEW_LINE INDENT total = 1 NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE INDENT total = total + ( ( x ** i ) / i ) NEW_LINE DEDENT return total NEW_LINE DEDENT
x = 2 NEW_LINE n = 5 NEW_LINE s = SUM ( x , n ) NEW_LINE print ( round ( s , 2 ) ) NEW_LINE
def check ( n ) : NEW_LINE
return 1162261467 % n == 0 NEW_LINE
n = 9 NEW_LINE if ( check ( n ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
import math NEW_LINE
def countDivisors ( n ) : NEW_LINE
count = 0 NEW_LINE
for i in range ( 1 , ( int ) ( math . sqrt ( n ) ) + 2 ) : NEW_LINE INDENT if ( n % i == 0 ) : NEW_LINE DEDENT
if ( n // i == i ) : NEW_LINE INDENT count = count + 1 NEW_LINE DEDENT else : NEW_LINE INDENT count = count + 2 NEW_LINE DEDENT if ( count % 2 == 0 ) : NEW_LINE print ( " Even " ) NEW_LINE else : NEW_LINE print ( " Odd " ) NEW_LINE
print ( " The ▁ count ▁ of ▁ divisor : ▁ " ) NEW_LINE countDivisors ( 10 ) NEW_LINE
def countSquares ( m , n ) : NEW_LINE
if ( n < m ) : NEW_LINE INDENT temp = m NEW_LINE m = n NEW_LINE n = temp NEW_LINE DEDENT
return ( ( m * ( m + 1 ) * ( 2 * m + 1 ) / 6 + ( n - m ) * m * ( m + 1 ) / 2 ) ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT m = 4 NEW_LINE n = 3 NEW_LINE print ( " Count ▁ of ▁ squares ▁ is ▁ " , countSquares ( m , n ) ) NEW_LINE DEDENT
def sum ( n ) : NEW_LINE INDENT i = 1 NEW_LINE s = 0.0 NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE INDENT s = s + 1 / i ; NEW_LINE DEDENT return s ; NEW_LINE DEDENT
n = 5 NEW_LINE print ( " Sum ▁ is " , round ( sum ( n ) , 6 ) ) NEW_LINE
def gcd ( a , b ) : ' NEW_LINE INDENT if ( b == 0 ) : NEW_LINE INDENT return a NEW_LINE DEDENT return gcd ( b , a % b ) NEW_LINE DEDENT
a = 98 NEW_LINE b = 56 NEW_LINE if ( gcd ( a , b ) ) : NEW_LINE INDENT print ( ' GCD ▁ of ' , a , ' and ' , b , ' is ' , gcd ( a , b ) ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( ' not ▁ found ' ) NEW_LINE DEDENT
def printArray ( arr , size ) : NEW_LINE INDENT for i in range ( size ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) ; NEW_LINE DEDENT print ( " " ) ; NEW_LINE return ; NEW_LINE DEDENT
def printSequencesRecur ( arr , n , k , index ) : NEW_LINE INDENT if ( k == 0 ) : NEW_LINE INDENT printArray ( arr , index ) ; NEW_LINE DEDENT if ( k > 0 ) : NEW_LINE INDENT for i in range ( 1 , n + 1 ) : NEW_LINE INDENT arr [ index ] = i ; NEW_LINE printSequencesRecur ( arr , n , k - 1 , index + 1 ) ; NEW_LINE DEDENT DEDENT DEDENT
def printSequences ( n , k ) : NEW_LINE INDENT arr = [ 0 ] * n ; NEW_LINE printSequencesRecur ( arr , n , k , 0 ) ; NEW_LINE return ; NEW_LINE DEDENT
n = 3 ; NEW_LINE k = 2 ; NEW_LINE printSequences ( n , k ) ; NEW_LINE
def isMultipleof5 ( n ) : NEW_LINE INDENT while ( n > 0 ) : NEW_LINE INDENT n = n - 5 NEW_LINE DEDENT if ( n == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT return 0 NEW_LINE DEDENT
i = 19 NEW_LINE if ( isMultipleof5 ( i ) == 1 ) : NEW_LINE INDENT print ( i , " is ▁ multiple ▁ of ▁ 5" ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( i , " is ▁ not ▁ a ▁ multiple ▁ of ▁ 5" ) NEW_LINE DEDENT
def countBits ( n ) : NEW_LINE INDENT count = 0 NEW_LINE while ( n ) : NEW_LINE INDENT count += 1 NEW_LINE n >>= 1 NEW_LINE DEDENT return count NEW_LINE DEDENT
i = 65 NEW_LINE print ( countBits ( i ) ) NEW_LINE
INT_MAX = 2147483647 NEW_LINE
def isKthBitSet ( x , k ) : NEW_LINE INDENT return 1 if ( x & ( 1 << ( k - 1 ) ) ) else 0 NEW_LINE DEDENT
def leftmostSetBit ( x ) : NEW_LINE INDENT count = 0 NEW_LINE while ( x ) : NEW_LINE INDENT count += 1 NEW_LINE x = x >> 1 NEW_LINE DEDENT return count NEW_LINE DEDENT
def isBinPalindrome ( x ) : NEW_LINE INDENT l = leftmostSetBit ( x ) NEW_LINE r = 1 NEW_LINE DEDENT
while ( l > r ) : NEW_LINE
if ( isKthBitSet ( x , l ) != isKthBitSet ( x , r ) ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT l -= 1 NEW_LINE r += 1 NEW_LINE return 1 NEW_LINE def findNthPalindrome ( n ) : NEW_LINE pal_count = 0 NEW_LINE
i = 0 NEW_LINE for i in range ( 1 , INT_MAX + 1 ) : NEW_LINE INDENT if ( isBinPalindrome ( i ) ) : NEW_LINE INDENT pal_count += 1 NEW_LINE DEDENT DEDENT
if ( pal_count == n ) : NEW_LINE INDENT break NEW_LINE DEDENT return i NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 9 NEW_LINE DEDENT
print ( findNthPalindrome ( n ) ) NEW_LINE
def temp_convert ( F1 , B1 , F2 , B2 , T ) : NEW_LINE
t2 = F2 + ( ( float ) ( B2 - F2 ) / ( B1 - F1 ) * ( T - F1 ) ) NEW_LINE return t2 NEW_LINE
F1 = 0 NEW_LINE B1 = 100 NEW_LINE F2 = 32 NEW_LINE B2 = 212 NEW_LINE T = 37 NEW_LINE print ( temp_convert ( F1 , B1 , F2 , B2 , T ) ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def maxDepth ( node ) : NEW_LINE INDENT if node is None : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT else : NEW_LINE DEDENT
lDepth = maxDepth ( node . left ) NEW_LINE rDepth = maxDepth ( node . right ) NEW_LINE
if ( lDepth > rDepth ) : NEW_LINE INDENT return lDepth + 1 NEW_LINE DEDENT else : NEW_LINE INDENT return rDepth + 1 NEW_LINE DEDENT
root = Node ( 1 ) NEW_LINE root . left = Node ( 2 ) NEW_LINE root . right = Node ( 3 ) NEW_LINE root . left . left = Node ( 4 ) NEW_LINE root . left . right = Node ( 5 ) NEW_LINE print ( " Height ▁ of ▁ tree ▁ is ▁ % d " % ( maxDepth ( root ) ) ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def isBalanced ( root ) : NEW_LINE
if root is None : NEW_LINE INDENT return True NEW_LINE DEDENT
lh = height ( root . left ) NEW_LINE rh = height ( root . right ) NEW_LINE if ( abs ( lh - rh ) <= 1 ) and isBalanced ( root . left ) is True and isBalanced ( root . right ) is True : NEW_LINE INDENT return True NEW_LINE DEDENT
return False NEW_LINE
def height ( root ) : NEW_LINE
if root is None : NEW_LINE INDENT return 0 NEW_LINE DEDENT
return max ( height ( root . left ) , height ( root . right ) ) + 1 NEW_LINE
root = Node ( 1 ) NEW_LINE root . left = Node ( 2 ) NEW_LINE root . right = Node ( 3 ) NEW_LINE root . left . left = Node ( 4 ) NEW_LINE root . left . right = Node ( 5 ) NEW_LINE root . left . left . left = Node ( 8 ) NEW_LINE if isBalanced ( root ) : NEW_LINE INDENT print ( " Tree ▁ is ▁ balanced " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Tree ▁ is ▁ not ▁ balanced " ) NEW_LINE DEDENT
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def height ( node ) : NEW_LINE
if node is None : NEW_LINE INDENT return 0 NEW_LINE DEDENT
return 1 + max ( height ( node . left ) , height ( node . right ) ) NEW_LINE
def diameter ( root ) : NEW_LINE
if root is None : NEW_LINE INDENT return 0 NEW_LINE DEDENT
lheight = height ( root . left ) NEW_LINE rheight = height ( root . right ) NEW_LINE
ldiameter = diameter ( root . left ) NEW_LINE rdiameter = diameter ( root . right ) NEW_LINE
return max ( lheight + rheight + 1 , max ( ldiameter , rdiameter ) ) NEW_LINE
root = Node ( 1 ) NEW_LINE root . left = Node ( 2 ) NEW_LINE root . right = Node ( 3 ) NEW_LINE root . left . left = Node ( 4 ) NEW_LINE root . left . right = Node ( 5 ) NEW_LINE
print ( diameter ( root ) ) NEW_LINE
def findpath ( N , a ) : NEW_LINE
if ( a [ 0 ] ) : NEW_LINE
print ( N + 1 ) NEW_LINE for i in range ( 1 , N + 1 , 1 ) : NEW_LINE INDENT print ( i , end = " ▁ " ) NEW_LINE DEDENT return NEW_LINE
for i in range ( N - 1 ) : NEW_LINE INDENT if ( a [ i ] == 0 and a [ i + 1 ] ) : NEW_LINE DEDENT
for j in range ( 1 , i + 1 , 1 ) : NEW_LINE INDENT print ( j , end = " ▁ " ) NEW_LINE DEDENT print ( N + 1 , end = " ▁ " ) ; NEW_LINE for j in range ( i + 1 , N + 1 , 1 ) : NEW_LINE INDENT print ( j , end = " ▁ " ) NEW_LINE DEDENT return NEW_LINE
for i in range ( 1 , N + 1 , 1 ) : NEW_LINE INDENT print ( i , end = " ▁ " ) NEW_LINE DEDENT print ( N + 1 , end = " ▁ " ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
N = 3 NEW_LINE arr = [ 0 , 1 , 0 ] NEW_LINE
findpath ( N , arr ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def depthOfOddLeafUtil ( root , level ) : NEW_LINE
if root is None : NEW_LINE INDENT return 0 NEW_LINE DEDENT
if root . left is None and root . right is None and level & 1 : NEW_LINE INDENT return level NEW_LINE DEDENT
return ( max ( depthOfOddLeafUtil ( root . left , level + 1 ) , depthOfOddLeafUtil ( root . right , level + 1 ) ) ) NEW_LINE
def depthOfOddLeaf ( root ) : NEW_LINE INDENT level = 1 NEW_LINE depth = 0 NEW_LINE return depthOfOddLeafUtil ( root , level ) NEW_LINE DEDENT
root = Node ( 1 ) NEW_LINE root . left = Node ( 2 ) NEW_LINE root . right = Node ( 3 ) NEW_LINE root . left . left = Node ( 4 ) NEW_LINE root . right . left = Node ( 5 ) NEW_LINE root . right . right = Node ( 6 ) NEW_LINE root . right . left . right = Node ( 7 ) NEW_LINE root . right . right . right = Node ( 8 ) NEW_LINE root . right . left . right . left = Node ( 9 ) NEW_LINE root . right . right . right . right = Node ( 10 ) NEW_LINE root . right . right . right . right . left = Node ( 11 ) NEW_LINE print " % d ▁ is ▁ the ▁ required ▁ depth " % ( depthOfOddLeaf ( root ) ) NEW_LINE
def printArr ( arr , n ) : NEW_LINE
arr . sort ( ) NEW_LINE
if ( arr [ 0 ] == arr [ n - 1 ] ) : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
else : NEW_LINE INDENT print ( " Yes " ) NEW_LINE for i in range ( n ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT print ( ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE
arr = [ 1 , 2 , 2 , 1 , 3 , 1 ] NEW_LINE N = len ( arr ) NEW_LINE
printArr ( arr , N ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def getMaxWidth ( root ) : NEW_LINE INDENT maxWidth = 0 NEW_LINE h = height ( root ) NEW_LINE DEDENT
for i in range ( 1 , h + 1 ) : NEW_LINE INDENT width = getWidth ( root , i ) NEW_LINE if ( width > maxWidth ) : NEW_LINE INDENT maxWidth = width NEW_LINE DEDENT DEDENT return maxWidth NEW_LINE
def getWidth ( root , level ) : NEW_LINE INDENT if root is None : NEW_LINE INDENT return 0 NEW_LINE DEDENT if level == 1 : NEW_LINE INDENT return 1 NEW_LINE DEDENT elif level > 1 : NEW_LINE INDENT return ( getWidth ( root . left , level - 1 ) + getWidth ( root . right , level - 1 ) ) NEW_LINE DEDENT DEDENT
def height ( node ) : NEW_LINE INDENT if node is None : NEW_LINE INDENT return 0 NEW_LINE DEDENT else : NEW_LINE DEDENT
lHeight = height ( node . left ) NEW_LINE rHeight = height ( node . right ) NEW_LINE
return ( lHeight + 1 ) if ( lHeight > rHeight ) else ( rHeight + 1 ) NEW_LINE
root = Node ( 1 ) NEW_LINE root . left = Node ( 2 ) NEW_LINE root . right = Node ( 3 ) NEW_LINE root . left . left = Node ( 4 ) NEW_LINE root . left . right = Node ( 5 ) NEW_LINE root . right . right = Node ( 8 ) NEW_LINE root . right . right . left = Node ( 6 ) NEW_LINE root . right . right . right = Node ( 7 ) NEW_LINE
print " Maximum ▁ width ▁ is ▁ % d " % ( getMaxWidth ( root ) ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def getMaxWidth ( root ) : NEW_LINE INDENT h = height ( root ) NEW_LINE DEDENT
count = [ 0 ] * h NEW_LINE level = 0 NEW_LINE
getMaxWidthRecur ( root , count , level ) NEW_LINE
return getMax ( count , h ) NEW_LINE
def getMaxWidthRecur ( root , count , level ) : NEW_LINE INDENT if root is not None : NEW_LINE INDENT count [ level ] += 1 NEW_LINE getMaxWidthRecur ( root . left , count , level + 1 ) NEW_LINE getMaxWidthRecur ( root . right , count , level + 1 ) NEW_LINE DEDENT DEDENT
def height ( node ) : NEW_LINE INDENT if node is None : NEW_LINE INDENT return 0 NEW_LINE DEDENT else : NEW_LINE DEDENT
lHeight = height ( node . left ) NEW_LINE rHeight = height ( node . right ) NEW_LINE
return ( lHeight + 1 ) if ( lHeight > rHeight ) else ( rHeight + 1 ) NEW_LINE
def getMax ( count , n ) : NEW_LINE INDENT max = count [ 0 ] NEW_LINE for i in range ( 1 , n ) : NEW_LINE INDENT if ( count [ i ] > max ) : NEW_LINE INDENT max = count [ i ] NEW_LINE DEDENT DEDENT return max NEW_LINE DEDENT
root = Node ( 1 ) NEW_LINE root . left = Node ( 2 ) NEW_LINE root . right = Node ( 3 ) NEW_LINE root . left . left = Node ( 4 ) NEW_LINE root . left . right = Node ( 5 ) NEW_LINE root . right . right = Node ( 8 ) NEW_LINE root . right . right . left = Node ( 6 ) NEW_LINE root . right . right . right = Node ( 7 ) NEW_LINE print " Maximum ▁ width ▁ is ▁ % d " % ( getMaxWidth ( root ) ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def getLeafCount ( node ) : NEW_LINE INDENT if node is None : NEW_LINE INDENT return 0 NEW_LINE DEDENT if ( node . left is None and node . right is None ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT else : NEW_LINE INDENT return getLeafCount ( node . left ) + getLeafCount ( node . right ) NEW_LINE DEDENT DEDENT
root = Node ( 1 ) NEW_LINE root . left = Node ( 2 ) NEW_LINE root . right = Node ( 3 ) NEW_LINE root . left . left = Node ( 4 ) NEW_LINE root . left . right = Node ( 5 ) NEW_LINE
print " Leaf ▁ count ▁ of ▁ the ▁ tree ▁ is ▁ % d " % ( getLeafCount ( root ) ) NEW_LINE
def findMin ( V ) : NEW_LINE
deno = [ 1 , 2 , 5 , 10 , 20 , 50 , 100 , 500 , 1000 ] NEW_LINE n = len ( deno ) NEW_LINE
ans = [ ] NEW_LINE
i = n - 1 NEW_LINE while ( i >= 0 ) : NEW_LINE
while ( V >= deno [ i ] ) : NEW_LINE INDENT V -= deno [ i ] NEW_LINE ans . append ( deno [ i ] ) NEW_LINE DEDENT i -= 1 NEW_LINE
for i in range ( len ( ans ) ) : NEW_LINE INDENT print ( ans [ i ] , end = " ▁ " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 93 NEW_LINE print ( " Following ▁ is ▁ minimal ▁ number " , " of ▁ change ▁ for " , n , " : ▁ " , end = " " ) NEW_LINE findMin ( n ) NEW_LINE DEDENT
class newnode : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE self . nextRight = None NEW_LINE DEDENT DEDENT
def getNextRight ( p ) : NEW_LINE INDENT temp = p . nextRight NEW_LINE DEDENT
while ( temp != None ) : NEW_LINE INDENT if ( temp . left != None ) : NEW_LINE INDENT return temp . left NEW_LINE DEDENT if ( temp . right != None ) : NEW_LINE INDENT return temp . right NEW_LINE DEDENT temp = temp . nextRight NEW_LINE DEDENT
return None NEW_LINE
def connect ( p ) : NEW_LINE INDENT temp = None NEW_LINE if ( not p ) : NEW_LINE INDENT return NEW_LINE DEDENT DEDENT
p . nextRight = None NEW_LINE
while ( p != None ) : NEW_LINE INDENT q = p NEW_LINE DEDENT
while ( q != None ) : NEW_LINE
if ( q . left ) : NEW_LINE
if ( q . right ) : NEW_LINE INDENT q . left . nextRight = q . right NEW_LINE DEDENT else : NEW_LINE INDENT q . left . nextRight = getNextRight ( q ) NEW_LINE DEDENT if ( q . right ) : NEW_LINE q . right . nextRight = getNextRight ( q ) NEW_LINE
q = q . nextRight NEW_LINE
if ( p . left ) : NEW_LINE INDENT p = p . left NEW_LINE DEDENT elif ( p . right ) : NEW_LINE INDENT p = p . right NEW_LINE DEDENT else : NEW_LINE INDENT p = getNextRight ( p ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE
root = newnode ( 10 ) NEW_LINE root . left = newnode ( 8 ) NEW_LINE root . right = newnode ( 2 ) NEW_LINE root . left . left = newnode ( 3 ) NEW_LINE root . right . right = newnode ( 90 ) NEW_LINE
connect ( root ) NEW_LINE
class newnode : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = self . right = self . nextRight = None NEW_LINE DEDENT DEDENT
def connect ( p ) : NEW_LINE
p . nextRight = None NEW_LINE
connectRecur ( p ) NEW_LINE
def connectRecur ( p ) : NEW_LINE
if ( not p ) : NEW_LINE INDENT return NEW_LINE DEDENT
if ( p . left ) : NEW_LINE INDENT p . left . nextRight = p . right NEW_LINE DEDENT
if ( p . right ) : NEW_LINE INDENT if p . nextRight : NEW_LINE INDENT p . right . nextRight = p . nextRight . left NEW_LINE DEDENT else : NEW_LINE INDENT p . right . nextRight = None NEW_LINE DEDENT DEDENT
connectRecur ( p . left ) NEW_LINE connectRecur ( p . right ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
root = newnode ( 10 ) NEW_LINE root . left = newnode ( 8 ) NEW_LINE root . right = newnode ( 2 ) NEW_LINE root . left . left = newnode ( 3 ) NEW_LINE
connect ( root ) NEW_LINE
print ( " Following ▁ are ▁ populated ▁ nextRight " , " pointers ▁ in ▁ the ▁ tree ▁ ( -1 ▁ is ▁ printed " , " if ▁ there ▁ is ▁ no ▁ nextRight ) " ) NEW_LINE print ( " nextRight ▁ of " , root . data , " is ▁ " , end = " " ) NEW_LINE if root . nextRight : NEW_LINE INDENT print ( root . nextRight . data ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( - 1 ) NEW_LINE DEDENT print ( " nextRight ▁ of " , root . left . data , " is ▁ " , end = " " ) NEW_LINE if root . left . nextRight : NEW_LINE INDENT print ( root . left . nextRight . data ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( - 1 ) NEW_LINE DEDENT print ( " nextRight ▁ of " , root . right . data , " is ▁ " , end = " " ) NEW_LINE if root . right . nextRight : NEW_LINE INDENT print ( root . right . nextRight . data ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( - 1 ) NEW_LINE DEDENT print ( " nextRight ▁ of " , root . left . left . data , " is ▁ " , end = " " ) NEW_LINE if root . left . left . nextRight : NEW_LINE INDENT print ( root . left . left . nextRight . data ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( - 1 ) NEW_LINE DEDENT
import sys NEW_LINE
def findMinInsertions ( str , l , h ) : NEW_LINE
if ( l > h ) : NEW_LINE INDENT return sys . maxsize NEW_LINE DEDENT if ( l == h ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT if ( l == h - 1 ) : NEW_LINE INDENT return 0 if ( str [ l ] == str [ h ] ) else 1 NEW_LINE DEDENT
if ( str [ l ] == str [ h ] ) : NEW_LINE INDENT return findMinInsertions ( str , l + 1 , h - 1 ) NEW_LINE DEDENT else : NEW_LINE INDENT return ( min ( findMinInsertions ( str , l , h - 1 ) , findMinInsertions ( str , l + 1 , h ) ) + 1 ) NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT str = " geeks " NEW_LINE print ( findMinInsertions ( str , 0 , len ( str ) - 1 ) ) NEW_LINE DEDENT
def max ( x , y ) : NEW_LINE INDENT if ( x > y ) : NEW_LINE INDENT return x NEW_LINE DEDENT return y NEW_LINE DEDENT
def lps ( seq , i , j ) : NEW_LINE
if ( i == j ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT
if ( seq [ i ] == seq [ j ] and i + 1 == j ) : NEW_LINE INDENT return 2 NEW_LINE DEDENT
if ( seq [ i ] == seq [ j ] ) : NEW_LINE INDENT return lps ( seq , i + 1 , j - 1 ) + 2 NEW_LINE DEDENT
return max ( lps ( seq , i , j - 1 ) , lps ( seq , i + 1 , j ) ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT seq = " GEEKSFORGEEKS " NEW_LINE n = len ( seq ) NEW_LINE print ( " The ▁ length ▁ of ▁ the ▁ LPS ▁ is " , lps ( seq , 0 , n - 1 ) ) NEW_LINE DEDENT
class newNode : NEW_LINE
def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT
def getLevelUtil ( node , data , level ) : NEW_LINE INDENT if ( node == None ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT if ( node . data == data ) : NEW_LINE INDENT return level NEW_LINE DEDENT downlevel = getLevelUtil ( node . left , data , level + 1 ) NEW_LINE if ( downlevel != 0 ) : NEW_LINE INDENT return downlevel NEW_LINE DEDENT downlevel = getLevelUtil ( node . right , data , level + 1 ) NEW_LINE return downlevel NEW_LINE DEDENT
def getLevel ( node , data ) : NEW_LINE INDENT return getLevelUtil ( node , data , 1 ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE
root = newNode ( 3 ) NEW_LINE root . left = newNode ( 2 ) NEW_LINE root . right = newNode ( 5 ) NEW_LINE root . left . left = newNode ( 1 ) NEW_LINE root . left . right = newNode ( 4 ) NEW_LINE for x in range ( 1 , 6 ) : NEW_LINE INDENT level = getLevel ( root , x ) NEW_LINE if ( level ) : NEW_LINE INDENT print ( " Level ▁ of " , x , " is " , getLevel ( root , x ) ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( x , " is ▁ not ▁ present ▁ in ▁ tree " ) NEW_LINE DEDENT DEDENT
NO_OF_CHARS = 256 NEW_LINE def getNextState ( pat , M , state , x ) : NEW_LINE
if state < M and x == ord ( pat [ state ] ) : NEW_LINE INDENT return state + 1 NEW_LINE DEDENT i = 0 NEW_LINE
for ns in range ( state , 0 , - 1 ) : NEW_LINE INDENT if ord ( pat [ ns - 1 ] ) == x : NEW_LINE INDENT while ( i < ns - 1 ) : NEW_LINE INDENT if pat [ i ] != pat [ state - ns + 1 + i ] : NEW_LINE INDENT break NEW_LINE DEDENT i += 1 NEW_LINE DEDENT if i == ns - 1 : NEW_LINE INDENT return ns NEW_LINE DEDENT DEDENT DEDENT return 0 NEW_LINE def computeTF ( pat , M ) : NEW_LINE
global NO_OF_CHARS NEW_LINE TF = [ [ 0 for i in range ( NO_OF_CHARS ) ] \ for _ in range ( M + 1 ) ] NEW_LINE for state in range ( M + 1 ) : NEW_LINE INDENT for x in range ( NO_OF_CHARS ) : NEW_LINE INDENT z = getNextState ( pat , M , state , x ) NEW_LINE TF [ state ] [ x ] = z NEW_LINE DEDENT DEDENT return TF NEW_LINE def search ( pat , txt ) : NEW_LINE
global NO_OF_CHARS NEW_LINE M = len ( pat ) NEW_LINE N = len ( txt ) NEW_LINE TF = computeTF ( pat , M ) NEW_LINE
state = 0 NEW_LINE for i in range ( N ) : NEW_LINE INDENT state = TF [ state ] [ ord ( txt [ i ] ) ] NEW_LINE if state == M : NEW_LINE INDENT print ( " Pattern ▁ found ▁ at ▁ index : ▁ { } " . format ( i - M + 1 ) ) NEW_LINE DEDENT DEDENT
def main ( ) : NEW_LINE INDENT txt = " AABAACAADAABAAABAA " NEW_LINE pat = " AABA " NEW_LINE search ( pat , txt ) NEW_LINE DEDENT if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT main ( ) NEW_LINE DEDENT
class Node : NEW_LINE
def __init__ ( self , key , lchild = None , rchild = None ) : NEW_LINE INDENT self . key = key NEW_LINE self . lchild = None NEW_LINE self . rchild = None NEW_LINE DEDENT
def findMirrorRec ( target , left , right ) : NEW_LINE
if left == None or right == None : NEW_LINE INDENT return None NEW_LINE DEDENT
if left . key == target : NEW_LINE INDENT return right . key NEW_LINE DEDENT if right . key == target : NEW_LINE INDENT return left . key NEW_LINE DEDENT
mirror_val = findMirrorRec ( target , left . lchild , right . rchild ) NEW_LINE if mirror_val != None : NEW_LINE INDENT return mirror_val NEW_LINE DEDENT
findMirrorRec ( target , left . rchild , right . lchild ) NEW_LINE
def findMirror ( root , target ) : NEW_LINE INDENT if root == None : NEW_LINE INDENT return None NEW_LINE DEDENT if root . key == target : NEW_LINE INDENT return target NEW_LINE DEDENT return findMirrorRec ( target , root . lchild , root . rchild ) NEW_LINE DEDENT
def main ( ) : NEW_LINE INDENT root = Node ( 1 ) NEW_LINE n1 = Node ( 2 ) NEW_LINE n2 = Node ( 3 ) NEW_LINE root . lchild = n1 NEW_LINE root . rchild = n2 NEW_LINE n3 = Node ( 4 ) NEW_LINE n4 = Node ( 5 ) NEW_LINE n5 = Node ( 6 ) NEW_LINE n1 . lchild = n3 NEW_LINE n2 . lchild = n4 NEW_LINE n2 . rchild = n5 NEW_LINE n6 = Node ( 7 ) NEW_LINE n7 = Node ( 8 ) NEW_LINE n8 = Node ( 9 ) NEW_LINE n3 . rchild = n6 NEW_LINE n4 . lchild = n7 NEW_LINE n4 . rchild = n8 NEW_LINE DEDENT
target = n3 . key NEW_LINE mirror = findMirror ( root , target ) NEW_LINE print ( " Mirror ▁ of ▁ node ▁ { } ▁ is ▁ node ▁ { } " . format ( target , mirror ) ) NEW_LINE if __name__ == ' _ _ main _ _ ' : NEW_LINE main ( ) NEW_LINE
from queue import Queue NEW_LINE
class newNode : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = self . right = None NEW_LINE DEDENT DEDENT
def iterativeSearch ( root , x ) : NEW_LINE
if ( root == None ) : NEW_LINE INDENT return False NEW_LINE DEDENT
q = Queue ( ) NEW_LINE
q . put ( root ) NEW_LINE
while ( q . empty ( ) == False ) : NEW_LINE
node = q . queue [ 0 ] NEW_LINE if ( node . data == x ) : NEW_LINE INDENT return True NEW_LINE DEDENT
q . get ( ) NEW_LINE if ( node . left != None ) : NEW_LINE INDENT q . put ( node . left ) NEW_LINE DEDENT if ( node . right != None ) : NEW_LINE INDENT q . put ( node . right ) NEW_LINE DEDENT return False NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT root = newNode ( 2 ) NEW_LINE root . left = newNode ( 7 ) NEW_LINE root . right = newNode ( 5 ) NEW_LINE root . left . right = newNode ( 6 ) NEW_LINE root . left . right . left = newNode ( 1 ) NEW_LINE root . left . right . right = newNode ( 11 ) NEW_LINE root . right . right = newNode ( 9 ) NEW_LINE root . right . right . left = newNode ( 4 ) NEW_LINE if iterativeSearch ( root , 6 ) : NEW_LINE INDENT print ( " Found " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Not ▁ Found " ) NEW_LINE DEDENT if iterativeSearch ( root , 12 ) : NEW_LINE INDENT print ( " Found " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Not ▁ Found " ) NEW_LINE DEDENT DEDENT
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE self . next = None NEW_LINE DEDENT DEDENT next = None NEW_LINE
def populateNext ( node ) : NEW_LINE
populateNextRecur ( node , next ) NEW_LINE
def populateNextRecur ( p , next_ref ) : NEW_LINE INDENT if ( p != None ) : NEW_LINE DEDENT
populateNextRecur ( p . right , next_ref ) NEW_LINE
p . next = next_ref NEW_LINE
next_ref = p NEW_LINE
populateNextRecur ( p . left , next_ref ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def printInorder ( root ) : NEW_LINE INDENT if root is not None : NEW_LINE INDENT printInorder ( root . left ) NEW_LINE print root . data , NEW_LINE printInorder ( root . right ) NEW_LINE DEDENT DEDENT
def RemoveHalfNodes ( root ) : NEW_LINE INDENT if root is None : NEW_LINE INDENT return None NEW_LINE DEDENT root . left = RemoveHalfNodes ( root . left ) NEW_LINE root . right = RemoveHalfNodes ( root . right ) NEW_LINE if root . left is None and root . right is None : NEW_LINE INDENT return root NEW_LINE DEDENT DEDENT
if root . left is None : NEW_LINE INDENT new_root = root . right NEW_LINE temp = root NEW_LINE root = None NEW_LINE del ( temp ) NEW_LINE return new_root NEW_LINE DEDENT
if root . right is None : NEW_LINE INDENT new_root = root . left NEW_LINE temp = root NEW_LINE root = None NEW_LINE del ( temp ) NEW_LINE return new_root NEW_LINE DEDENT return root NEW_LINE
root = Node ( 2 ) NEW_LINE root . left = Node ( 7 ) NEW_LINE root . right = Node ( 5 ) NEW_LINE root . left . right = Node ( 6 ) NEW_LINE root . left . right . left = Node ( 1 ) NEW_LINE root . left . right . right = Node ( 11 ) NEW_LINE root . right . right = Node ( 9 ) NEW_LINE root . right . right . left = Node ( 4 ) NEW_LINE print " Inorder ▁ traversal ▁ of ▁ given ▁ tree " NEW_LINE printInorder ( root ) NEW_LINE NewRoot = RemoveHalfNodes ( root ) NEW_LINE print   " NEW_LINE Inorder traversal of the modified tree " NEW_LINE printInorder ( NewRoot ) NEW_LINE
def printSubstrings ( string , n ) : NEW_LINE
for i in range ( n ) : NEW_LINE
for j in range ( i , n ) : NEW_LINE
for k in range ( i , ( j + 1 ) ) : NEW_LINE INDENT print ( string [ k ] , end = " " ) NEW_LINE DEDENT
print ( ) NEW_LINE
string = " abcd " NEW_LINE
printSubstrings ( string , len ( string ) ) NEW_LINE
N = 9 NEW_LINE
def printing ( arr ) : NEW_LINE INDENT for i in range ( N ) : NEW_LINE INDENT for j in range ( N ) : NEW_LINE INDENT print ( arr [ i ] [ j ] , end = " ▁ " ) NEW_LINE DEDENT print ( ) NEW_LINE DEDENT DEDENT
def isSafe ( grid , row , col , num ) : NEW_LINE
for x in range ( 9 ) : NEW_LINE INDENT if grid [ row ] [ x ] == num : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT
for x in range ( 9 ) : NEW_LINE INDENT if grid [ x ] [ col ] == num : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT
startRow = row - row % 3 NEW_LINE startCol = col - col % 3 NEW_LINE for i in range ( 3 ) : NEW_LINE INDENT for j in range ( 3 ) : NEW_LINE INDENT if grid [ i + startRow ] [ j + startCol ] == num : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT DEDENT return True NEW_LINE
def solveSuduko ( grid , row , col ) : NEW_LINE
if ( row == N - 1 and col == N ) : NEW_LINE INDENT return True NEW_LINE DEDENT
if col == N : NEW_LINE INDENT row += 1 NEW_LINE col = 0 NEW_LINE DEDENT
if grid [ row ] [ col ] > 0 : NEW_LINE INDENT return solveSuduko ( grid , row , col + 1 ) NEW_LINE DEDENT for num in range ( 1 , N + 1 , 1 ) : NEW_LINE
if isSafe ( grid , row , col , num ) : NEW_LINE
grid [ row ] [ col ] = num NEW_LINE
if solveSuduko ( grid , row , col + 1 ) : NEW_LINE INDENT return True NEW_LINE DEDENT
grid [ row ] [ col ] = 0 NEW_LINE return False NEW_LINE
grid = [ [ 3 , 0 , 6 , 5 , 0 , 8 , 4 , 0 , 0 ] , [ 5 , 2 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ] , [ 0 , 8 , 7 , 0 , 0 , 0 , 0 , 3 , 1 ] , [ 0 , 0 , 3 , 0 , 1 , 0 , 0 , 8 , 0 ] , [ 9 , 0 , 0 , 8 , 6 , 3 , 0 , 0 , 5 ] , [ 0 , 5 , 0 , 0 , 9 , 0 , 6 , 0 , 0 ] , [ 1 , 3 , 0 , 0 , 0 , 0 , 2 , 5 , 0 ] , [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 7 , 4 ] , [ 0 , 0 , 5 , 2 , 0 , 6 , 3 , 0 , 0 ] ] NEW_LINE if ( solveSuduko ( grid , 0 , 0 ) ) : NEW_LINE INDENT printing ( grid ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " no ▁ solution ▁ exists ▁ " ) NEW_LINE DEDENT
def printPairs ( arr , arr_size , sum ) : NEW_LINE INDENT s = set ( ) NEW_LINE for i in range ( 0 , arr_size ) : NEW_LINE INDENT temp = sum - arr [ i ] NEW_LINE DEDENT DEDENT
if ( temp in s ) : NEW_LINE INDENT print " Pair ▁ with ▁ given ▁ sum ▁ " + str ( sum ) + NEW_LINE DEDENT " ▁ is ▁ ( " + str ( arr [ i ] ) + " , ▁ " + str ( temp ) + " ) " NEW_LINE s . add ( arr [ i ] ) NEW_LINE
A = [ 1 , 4 , 45 , 6 , 10 , 8 ] NEW_LINE n = 16 NEW_LINE printPairs ( A , len ( A ) , n ) NEW_LINE
def exponentMod ( A , B , C ) : NEW_LINE
if ( A == 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT if ( B == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT
y = 0 NEW_LINE if ( B % 2 == 0 ) : NEW_LINE INDENT y = exponentMod ( A , B / 2 , C ) NEW_LINE y = ( y * y ) % C NEW_LINE DEDENT
else : NEW_LINE INDENT y = A % C NEW_LINE y = ( y * exponentMod ( A , B - 1 , C ) % C ) % C NEW_LINE DEDENT return ( ( y + C ) % C ) NEW_LINE
A = 2 NEW_LINE B = 5 NEW_LINE C = 13 NEW_LINE print ( " Power ▁ is " , exponentMod ( A , B , C ) ) NEW_LINE
def power ( x , y ) : NEW_LINE
res = 1 NEW_LINE while ( y > 0 ) : NEW_LINE
if ( ( y & 1 ) != 0 ) : NEW_LINE INDENT res = res * x NEW_LINE DEDENT
y = y >> 1 NEW_LINE
x = x * x NEW_LINE INDENT return res NEW_LINE DEDENT
import sys NEW_LINE
def eggDrop ( n , k ) : NEW_LINE
if ( k == 1 or k == 0 ) : NEW_LINE INDENT return k NEW_LINE DEDENT
if ( n == 1 ) : NEW_LINE INDENT return k NEW_LINE DEDENT min = sys . maxsize NEW_LINE
for x in range ( 1 , k + 1 ) : NEW_LINE INDENT res = max ( eggDrop ( n - 1 , x - 1 ) , eggDrop ( n , k - x ) ) NEW_LINE if ( res < min ) : NEW_LINE INDENT min = res NEW_LINE DEDENT DEDENT return min + 1 NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 2 NEW_LINE k = 10 NEW_LINE print ( " Minimum ▁ number ▁ of ▁ trials ▁ in ▁ worst ▁ case ▁ with " , n , " eggs ▁ and " , k , " floors ▁ is " , eggDrop ( n , k ) ) NEW_LINE DEDENT
class Node : NEW_LINE INDENT def __init__ ( self , key ) : NEW_LINE INDENT self . left = None NEW_LINE self . right = None NEW_LINE self . val = key NEW_LINE DEDENT DEDENT
root = Node ( 1 ) NEW_LINE root . left = Node ( 2 ) ; NEW_LINE root . right = Node ( 3 ) ; NEW_LINE root . left . left = Node ( 4 ) ; NEW_LINE
class newNode : NEW_LINE
def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = self . right = None NEW_LINE DEDENT
def findMax ( root ) : NEW_LINE
if ( root == None ) : NEW_LINE INDENT return float ( ' - inf ' ) NEW_LINE DEDENT
res = root . data NEW_LINE lres = findMax ( root . left ) NEW_LINE rres = findMax ( root . right ) NEW_LINE if ( lres > res ) : NEW_LINE INDENT res = lres NEW_LINE DEDENT if ( rres > res ) : NEW_LINE INDENT res = rres NEW_LINE DEDENT return res NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT root = newNode ( 2 ) NEW_LINE root . left = newNode ( 7 ) NEW_LINE root . right = newNode ( 5 ) NEW_LINE root . left . right = newNode ( 6 ) NEW_LINE root . left . right . left = newNode ( 1 ) NEW_LINE root . left . right . right = newNode ( 11 ) NEW_LINE root . right . right = newNode ( 9 ) NEW_LINE root . right . right . left = newNode ( 4 ) NEW_LINE DEDENT
print ( " Maximum ▁ element ▁ is " , findMax ( root ) ) NEW_LINE
def find_min_in_BT ( root ) : NEW_LINE INDENT if root is None : NEW_LINE INDENT return float ( ' inf ' ) NEW_LINE DEDENT res = root . data NEW_LINE lres = find_min_in_BT ( root . leftChild ) NEW_LINE rres = find_min_in_BT ( root . rightChild ) NEW_LINE if lres < res : NEW_LINE INDENT res = lres NEW_LINE DEDENT if rres < res : NEW_LINE INDENT res = rres NEW_LINE DEDENT return res NEW_LINE DEDENT
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def extractLeafList ( root ) : NEW_LINE INDENT if root is None : NEW_LINE INDENT return None NEW_LINE DEDENT if root . left is None and root . right is None : NEW_LINE INDENT root . right = extractLeafList . head NEW_LINE if extractLeafList . head is not None : NEW_LINE INDENT extractLeafList . head . left = root NEW_LINE DEDENT extractLeafList . head = root NEW_LINE return None NEW_LINE DEDENT root . right = extractLeafList ( root . right ) NEW_LINE root . left = extractLeafList ( root . left ) NEW_LINE return root NEW_LINE DEDENT
def printInorder ( root ) : NEW_LINE INDENT if root is not None : NEW_LINE INDENT printInorder ( root . left ) NEW_LINE print root . data , NEW_LINE printInorder ( root . right ) NEW_LINE DEDENT DEDENT
def printList ( head ) : NEW_LINE INDENT while ( head ) : NEW_LINE INDENT if head . data is not None : NEW_LINE INDENT print head . data , NEW_LINE DEDENT head = head . right NEW_LINE DEDENT DEDENT
extractLeafList . head = Node ( None ) NEW_LINE root = Node ( 1 ) NEW_LINE root . left = Node ( 2 ) NEW_LINE root . right = Node ( 3 ) NEW_LINE root . left . left = Node ( 4 ) NEW_LINE root . left . right = Node ( 5 ) NEW_LINE root . right . right = Node ( 6 ) NEW_LINE root . left . left . left = Node ( 7 ) NEW_LINE root . left . left . right = Node ( 8 ) NEW_LINE root . right . right . left = Node ( 9 ) NEW_LINE root . right . right . right = Node ( 10 ) NEW_LINE print " Inorder ▁ traversal ▁ of ▁ given ▁ tree ▁ is : " NEW_LINE printInorder ( root ) NEW_LINE root = extractLeafList ( root ) NEW_LINE print   " NEW_LINE Extract Double Linked List is : " NEW_LINE printList ( extractLeafList . head ) NEW_LINE print   " NEW_LINE Inorder traversal of modified tree is : " NEW_LINE printInorder ( root ) NEW_LINE
count = [ 0 ] NEW_LINE
class newNode : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE self . visited = False NEW_LINE DEDENT DEDENT
def NthInorder ( node , n ) : NEW_LINE INDENT if ( node == None ) : NEW_LINE INDENT return NEW_LINE DEDENT if ( count [ 0 ] <= n ) : NEW_LINE DEDENT
NthInorder ( node . left , n ) NEW_LINE count [ 0 ] += 1 NEW_LINE
if ( count [ 0 ] == n ) : NEW_LINE INDENT print ( node . data , end = " ▁ " ) NEW_LINE DEDENT
NthInorder ( node . right , n ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT root = newNode ( 10 ) NEW_LINE root . left = newNode ( 20 ) NEW_LINE root . right = newNode ( 30 ) NEW_LINE root . left . left = newNode ( 40 ) NEW_LINE root . left . right = newNode ( 50 ) NEW_LINE n = 4 NEW_LINE NthInorder ( root , n ) NEW_LINE DEDENT
def quickSort ( arr , low , high ) : NEW_LINE INDENT if ( low < high ) : NEW_LINE DEDENT
pi = partition ( arr , low , high ) NEW_LINE
quickSort ( arr , low , pi - 1 ) NEW_LINE quickSort ( arr , pi + 1 , high ) NEW_LINE
def countNumberOfStrings ( s ) : NEW_LINE
length = len ( s ) NEW_LINE
n = length - 1 NEW_LINE
count = 2 ** n NEW_LINE return count NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT S = " ABCD " NEW_LINE print ( countNumberOfStrings ( S ) ) NEW_LINE DEDENT
def makeArraySumEqual ( a , N ) : NEW_LINE
count_0 = 0 NEW_LINE count_1 = 0 NEW_LINE
odd_sum = 0 NEW_LINE even_sum = 0 NEW_LINE for i in range ( N ) : NEW_LINE
if ( a [ i ] == 0 ) : NEW_LINE INDENT count_0 += 1 NEW_LINE DEDENT
else : NEW_LINE INDENT count_1 += 1 NEW_LINE DEDENT
if ( ( i + 1 ) % 2 == 0 ) : NEW_LINE INDENT even_sum += a [ i ] NEW_LINE DEDENT elif ( ( i + 1 ) % 2 > 0 ) : NEW_LINE INDENT odd_sum += a [ i ] NEW_LINE DEDENT
if ( odd_sum == even_sum ) : NEW_LINE
for i in range ( N ) : NEW_LINE INDENT print ( a [ i ] , end = " ▁ " ) NEW_LINE DEDENT
else : NEW_LINE INDENT if ( count_0 >= N / 2 ) : NEW_LINE DEDENT
for i in range ( count_0 ) : NEW_LINE INDENT print ( "0" , end = " ▁ " ) NEW_LINE DEDENT else : NEW_LINE
is_Odd = count_1 % 2 NEW_LINE
count_1 -= is_Odd NEW_LINE
for i in range ( count_1 ) : NEW_LINE INDENT print ( "1" , end = " ▁ " ) NEW_LINE DEDENT
arr = [ 1 , 1 , 1 , 0 ] NEW_LINE N = len ( arr ) NEW_LINE
makeArraySumEqual ( arr , N ) NEW_LINE
def countDigitSum ( N , K ) : NEW_LINE
l = pow ( 10 , N - 1 ) ; NEW_LINE r = pow ( 10 , N ) - 1 ; NEW_LINE count = 0 ; NEW_LINE for i in range ( 1 , r + 1 ) : NEW_LINE INDENT num = i ; NEW_LINE DEDENT
digits = [ 0 ] * ( N ) ; NEW_LINE for j in range ( N - 1 , 0 , - 1 ) : NEW_LINE INDENT digits [ j ] = num % 10 ; NEW_LINE num //= 10 ; NEW_LINE DEDENT sum = 0 ; NEW_LINE flag = 0 ; NEW_LINE
for j in range ( 0 , K ) : NEW_LINE INDENT sum += digits [ j ] ; NEW_LINE DEDENT
for j in range ( K , N ) : NEW_LINE INDENT if ( sum - digits [ j - K ] + digits [ j ] != sum ) : NEW_LINE INDENT flag = 1 ; NEW_LINE break ; NEW_LINE DEDENT DEDENT if ( flag == 0 ) : NEW_LINE INDENT count += 1 ; NEW_LINE DEDENT return count ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
N = 2 ; NEW_LINE K = 1 ; NEW_LINE print ( countDigitSum ( N , K ) ) ; NEW_LINE
def findpath ( N , a ) : NEW_LINE
if ( a [ 0 ] ) : NEW_LINE
print ( N + 1 ) NEW_LINE for i in range ( 1 , N + 1 , 1 ) : NEW_LINE INDENT print ( i , end = " ▁ " ) NEW_LINE DEDENT return NEW_LINE
for i in range ( N - 1 ) : NEW_LINE INDENT if ( a [ i ] == 0 and a [ i + 1 ] ) : NEW_LINE DEDENT
for j in range ( 1 , i + 1 , 1 ) : NEW_LINE INDENT print ( j , end = " ▁ " ) NEW_LINE DEDENT print ( N + 1 , end = " ▁ " ) ; NEW_LINE for j in range ( i + 1 , N + 1 , 1 ) : NEW_LINE INDENT print ( j , end = " ▁ " ) NEW_LINE DEDENT return NEW_LINE
for i in range ( 1 , N + 1 , 1 ) : NEW_LINE INDENT print ( i , end = " ▁ " ) NEW_LINE DEDENT print ( N + 1 , end = " ▁ " ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
N = 3 NEW_LINE arr = [ 0 , 1 , 0 ] NEW_LINE
findpath ( N , arr ) NEW_LINE
def printknapSack ( W , wt , val , n ) : NEW_LINE INDENT K = [ [ 0 for w in range ( W + 1 ) ] for i in range ( n + 1 ) ] NEW_LINE DEDENT
for i in range ( n + 1 ) : NEW_LINE INDENT for w in range ( W + 1 ) : NEW_LINE INDENT if i == 0 or w == 0 : NEW_LINE INDENT K [ i ] [ w ] = 0 NEW_LINE DEDENT elif wt [ i - 1 ] <= w : NEW_LINE INDENT K [ i ] [ w ] = max ( val [ i - 1 ] + K [ i - 1 ] [ w - wt [ i - 1 ] ] , K [ i - 1 ] [ w ] ) NEW_LINE DEDENT else : NEW_LINE INDENT K [ i ] [ w ] = K [ i - 1 ] [ w ] NEW_LINE DEDENT DEDENT DEDENT
res = K [ n ] [ W ] NEW_LINE print ( res ) NEW_LINE w = W NEW_LINE for i in range ( n , 0 , - 1 ) : NEW_LINE INDENT if res <= 0 : NEW_LINE INDENT break NEW_LINE DEDENT DEDENT
if res == K [ i - 1 ] [ w ] : NEW_LINE INDENT continue NEW_LINE DEDENT else : NEW_LINE
print ( wt [ i - 1 ] ) NEW_LINE
res = res - val [ i - 1 ] NEW_LINE w = w - wt [ i - 1 ] NEW_LINE
val = [ 60 , 100 , 120 ] NEW_LINE wt = [ 10 , 20 , 30 ] NEW_LINE W = 50 NEW_LINE n = len ( val ) NEW_LINE printknapSack ( W , wt , val , n ) NEW_LINE
def optCost ( freq , i , j ) : NEW_LINE
return 0 NEW_LINE
if j == i : NEW_LINE INDENT return freq [ i ] NEW_LINE DEDENT
fsum = Sum ( freq , i , j ) NEW_LINE
Min = 999999999999 NEW_LINE
for r in range ( i , j + 1 ) : NEW_LINE INDENT cost = ( optCost ( freq , i , r - 1 ) + optCost ( freq , r + 1 , j ) ) NEW_LINE if cost < Min : NEW_LINE INDENT Min = cost NEW_LINE DEDENT DEDENT
return Min + fsum NEW_LINE
def optimalSearchTree ( keys , freq , n ) : NEW_LINE
return optCost ( freq , 0 , n - 1 ) NEW_LINE
def Sum ( freq , i , j ) : NEW_LINE INDENT s = 0 NEW_LINE for k in range ( i , j + 1 ) : NEW_LINE INDENT s += freq [ k ] NEW_LINE DEDENT return s NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT keys = [ 10 , 12 , 20 ] NEW_LINE freq = [ 34 , 8 , 50 ] NEW_LINE n = len ( keys ) NEW_LINE print ( " Cost ▁ of ▁ Optimal ▁ BST ▁ is " , optimalSearchTree ( keys , freq , n ) ) NEW_LINE DEDENT
INF = 2147483647 NEW_LINE def printSolution ( p , n ) : NEW_LINE INDENT k = 0 NEW_LINE if p [ n ] == 1 : NEW_LINE INDENT k = 1 NEW_LINE DEDENT else : NEW_LINE INDENT k = printSolution ( p , p [ n ] - 1 ) + 1 NEW_LINE DEDENT print ( ' Line ▁ number ▁ ' , k , ' : ▁ From ▁ word ▁ no . ▁ ' , p [ n ] , ' to ▁ ' , n ) NEW_LINE return k NEW_LINE DEDENT
def solveWordWrap ( l , n , M ) : NEW_LINE
extras = [ [ 0 for i in range ( n + 1 ) ] for i in range ( n + 1 ) ] NEW_LINE
lc = [ [ 0 for i in range ( n + 1 ) ] for i in range ( n + 1 ) ] NEW_LINE
c = [ 0 for i in range ( n + 1 ) ] NEW_LINE
p = [ 0 for i in range ( n + 1 ) ] NEW_LINE
for i in range ( n + 1 ) : NEW_LINE INDENT extras [ i ] [ i ] = M - l [ i - 1 ] NEW_LINE for j in range ( i + 1 , n + 1 ) : NEW_LINE INDENT extras [ i ] [ j ] = ( extras [ i ] [ j - 1 ] - l [ j - 1 ] - 1 ) NEW_LINE DEDENT DEDENT
for i in range ( n + 1 ) : NEW_LINE INDENT for j in range ( i , n + 1 ) : NEW_LINE INDENT if extras [ i ] [ j ] < 0 : NEW_LINE INDENT lc [ i ] [ j ] = INF ; NEW_LINE DEDENT elif j == n and extras [ i ] [ j ] >= 0 : NEW_LINE INDENT lc [ i ] [ j ] = 0 NEW_LINE DEDENT else : NEW_LINE INDENT lc [ i ] [ j ] = ( extras [ i ] [ j ] * extras [ i ] [ j ] ) NEW_LINE DEDENT DEDENT DEDENT
c [ 0 ] = 0 NEW_LINE for j in range ( 1 , n + 1 ) : NEW_LINE INDENT c [ j ] = INF NEW_LINE for i in range ( 1 , j + 1 ) : NEW_LINE INDENT if ( c [ i - 1 ] != INF and lc [ i ] [ j ] != INF and ( ( c [ i - 1 ] + lc [ i ] [ j ] ) < c [ j ] ) ) : NEW_LINE INDENT c [ j ] = c [ i - 1 ] + lc [ i ] [ j ] NEW_LINE p [ j ] = i NEW_LINE DEDENT DEDENT DEDENT printSolution ( p , n ) NEW_LINE
l = [ 3 , 2 , 2 , 5 ] NEW_LINE n = len ( l ) NEW_LINE M = 6 NEW_LINE solveWordWrap ( l , n , M ) NEW_LINE
INT_MAX = 32767 NEW_LINE
def eggDrop ( n , k ) : NEW_LINE
eggFloor = [ [ 0 for x in range ( k + 1 ) ] for x in range ( n + 1 ) ] NEW_LINE
for i in range ( 1 , n + 1 ) : NEW_LINE INDENT eggFloor [ i ] [ 1 ] = 1 NEW_LINE eggFloor [ i ] [ 0 ] = 0 NEW_LINE DEDENT
for j in range ( 1 , k + 1 ) : NEW_LINE INDENT eggFloor [ 1 ] [ j ] = j NEW_LINE DEDENT
for i in range ( 2 , n + 1 ) : NEW_LINE INDENT for j in range ( 2 , k + 1 ) : NEW_LINE INDENT eggFloor [ i ] [ j ] = INT_MAX NEW_LINE for x in range ( 1 , j + 1 ) : NEW_LINE INDENT res = 1 + max ( eggFloor [ i - 1 ] [ x - 1 ] , eggFloor [ i ] [ j - x ] ) NEW_LINE if res < eggFloor [ i ] [ j ] : NEW_LINE INDENT eggFloor [ i ] [ j ] = res NEW_LINE DEDENT DEDENT DEDENT DEDENT
return eggFloor [ n ] [ k ] NEW_LINE
n = 2 NEW_LINE k = 36 NEW_LINE print ( " Minimum ▁ number ▁ of ▁ trials ▁ in ▁ worst ▁ case ▁ with " + str ( n ) + " eggs ▁ and ▁ " + str ( k ) + " ▁ floors ▁ is ▁ " + str ( eggDrop ( n , k ) ) ) NEW_LINE
def knapSack ( W , wt , val , n ) : NEW_LINE
if n == 0 or W == 0 : NEW_LINE INDENT return 0 NEW_LINE DEDENT
if ( wt [ n - 1 ] > W ) : NEW_LINE INDENT return knapSack ( W , wt , val , n - 1 ) NEW_LINE DEDENT
else : NEW_LINE INDENT return max ( val [ n - 1 ] + knapSack ( W - wt [ n - 1 ] , wt , val , n - 1 ) , knapSack ( W , wt , val , n - 1 ) ) NEW_LINE DEDENT
val = [ 60 , 100 , 120 ] NEW_LINE wt = [ 10 , 20 , 30 ] NEW_LINE W = 50 NEW_LINE n = len ( val ) NEW_LINE print knapSack ( W , wt , val , n ) NEW_LINE
global maximum NEW_LINE def _lis ( arr , n ) : NEW_LINE
if n == 1 : NEW_LINE INDENT return 1 NEW_LINE DEDENT
maxEndingHere = 1 NEW_LINE
for i in xrange ( 1 , n ) : NEW_LINE INDENT res = _lis ( arr , i ) NEW_LINE if arr [ i - 1 ] < arr [ n - 1 ] and res + 1 > maxEndingHere : NEW_LINE INDENT maxEndingHere = res + 1 NEW_LINE DEDENT DEDENT
maximum = max ( maximum , maxEndingHere ) NEW_LINE
return maxEndingHere NEW_LINE
def lis ( arr ) : NEW_LINE
n = len ( arr ) NEW_LINE maximum = 1 NEW_LINE
_lis ( arr , n ) NEW_LINE
return maximum NEW_LINE
arr = [ 10 , 22 , 9 , 33 , 21 , 50 , 41 , 60 ] NEW_LINE n = len ( arr ) NEW_LINE print " Length ▁ of ▁ lis ▁ is ▁ " , lis ( arr ) NEW_LINE
d = 256 NEW_LINE
def search ( pat , txt , q ) : NEW_LINE INDENT M = len ( pat ) NEW_LINE N = len ( txt ) NEW_LINE i = 0 NEW_LINE j = 0 NEW_LINE DEDENT
h = 1 NEW_LINE
for i in xrange ( M - 1 ) : NEW_LINE INDENT h = ( h * d ) % q NEW_LINE DEDENT
for i in xrange ( M ) : NEW_LINE INDENT p = ( d * p + ord ( pat [ i ] ) ) % q NEW_LINE t = ( d * t + ord ( txt [ i ] ) ) % q NEW_LINE DEDENT
for i in xrange ( N - M + 1 ) : NEW_LINE
if p == t : NEW_LINE
for j in xrange ( M ) : NEW_LINE INDENT if txt [ i + j ] != pat [ j ] : NEW_LINE INDENT break NEW_LINE DEDENT else : j += 1 NEW_LINE DEDENT
if j == M : NEW_LINE INDENT print " Pattern ▁ found ▁ at ▁ index ▁ " + str ( i ) NEW_LINE DEDENT
if i < N - M : NEW_LINE INDENT t = ( d * ( t - ord ( txt [ i ] ) * h ) + ord ( txt [ i + M ] ) ) % q NEW_LINE DEDENT
if t < 0 : NEW_LINE INDENT t = t + q NEW_LINE DEDENT
txt = " GEEKS ▁ FOR ▁ GEEKS " NEW_LINE pat = " GEEK " NEW_LINE
q = 101 NEW_LINE
search ( pat , txt , q ) NEW_LINE
n = 8 NEW_LINE
def isSafe ( x , y , board ) : NEW_LINE INDENT if ( x >= 0 and y >= 0 and x < n and y < n and board [ x ] [ y ] == - 1 ) : NEW_LINE INDENT return True NEW_LINE DEDENT return False NEW_LINE DEDENT
def printSolution ( n , board ) : NEW_LINE INDENT for i in range ( n ) : NEW_LINE INDENT for j in range ( n ) : NEW_LINE INDENT print ( board [ i ] [ j ] , end = ' ▁ ' ) NEW_LINE DEDENT print ( ) NEW_LINE DEDENT DEDENT
def solveKT ( n ) : NEW_LINE
board = [ [ - 1 for i in range ( n ) ] for i in range ( n ) ] NEW_LINE
move_x = [ 2 , 1 , - 1 , - 2 , - 2 , - 1 , 1 , 2 ] NEW_LINE move_y = [ 1 , 2 , 2 , 1 , - 1 , - 2 , - 2 , - 1 ] NEW_LINE
board [ 0 ] [ 0 ] = 0 NEW_LINE
pos = 1 NEW_LINE if ( not solveKTUtil ( n , board , 0 , 0 , move_x , move_y , pos ) ) : NEW_LINE INDENT print ( " Solution ▁ does ▁ not ▁ exist " ) NEW_LINE DEDENT else : NEW_LINE INDENT printSolution ( n , board ) NEW_LINE DEDENT
def solveKTUtil ( n , board , curr_x , curr_y , move_x , move_y , pos ) : NEW_LINE INDENT if ( pos == n ** 2 ) : NEW_LINE INDENT return True NEW_LINE DEDENT DEDENT
for i in range ( 8 ) : NEW_LINE INDENT new_x = curr_x + move_x [ i ] NEW_LINE new_y = curr_y + move_y [ i ] NEW_LINE if ( isSafe ( new_x , new_y , board ) ) : NEW_LINE INDENT board [ new_x ] [ new_y ] = pos NEW_LINE if ( solveKTUtil ( n , board , new_x , new_y , move_x , move_y , pos + 1 ) ) : NEW_LINE INDENT return True NEW_LINE DEDENT DEDENT DEDENT
board [ new_x ] [ new_y ] = - 1 NEW_LINE return False NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
solveKT ( n ) NEW_LINE
def printSolution ( color ) : NEW_LINE INDENT print ( " Solution ▁ Exists : " " ▁ Following ▁ are ▁ the ▁ assigned ▁ colors ▁ " ) NEW_LINE for i in range ( 4 ) : NEW_LINE INDENT print ( color [ i ] , end = " ▁ " ) NEW_LINE DEDENT DEDENT
def isSafe ( graph , color ) : NEW_LINE
for i in range ( 4 ) : NEW_LINE INDENT for j in range ( i + 1 , 4 ) : NEW_LINE INDENT if ( graph [ i ] [ j ] and color [ j ] == color [ i ] ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT DEDENT return True NEW_LINE
def graphColoring ( graph , m , i , color ) : NEW_LINE
if ( i == 4 ) : NEW_LINE
if ( isSafe ( graph , color ) ) : NEW_LINE
printSolution ( color ) NEW_LINE return True NEW_LINE return False NEW_LINE
for j in range ( 1 , m + 1 ) : NEW_LINE INDENT color [ i ] = j NEW_LINE DEDENT
if ( graphColoring ( graph , m , i + 1 , color ) ) : NEW_LINE INDENT return True NEW_LINE DEDENT color [ i ] = 0 NEW_LINE return False NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
graph = [ [ 0 , 1 , 1 , 1 ] , [ 1 , 0 , 1 , 0 ] , [ 1 , 1 , 0 , 1 ] , [ 1 , 0 , 1 , 0 ] , ] NEW_LINE
m = 3 NEW_LINE
color = [ 0 for i in range ( 4 ) ] NEW_LINE if ( not graphColoring ( graph , m , 0 , color ) ) : NEW_LINE INDENT print ( " Solution ▁ does ▁ not ▁ exist " ) NEW_LINE DEDENT
from math import log NEW_LINE
def prevPowerofK ( n , k ) : NEW_LINE INDENT p = ( int ) ( log ( n ) / log ( k ) ) ; NEW_LINE return pow ( k , p ) ; NEW_LINE DEDENT
def nextPowerOfK ( n , k ) : NEW_LINE INDENT return prevPowerofK ( n , k ) * k ; NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT N = 7 NEW_LINE K = 2 NEW_LINE print ( prevPowerofK ( N , K ) , end = " ▁ " ) NEW_LINE print ( nextPowerOfK ( N , K ) ) NEW_LINE DEDENT
def gcd ( a , b ) : NEW_LINE INDENT if ( b == 0 ) : NEW_LINE INDENT return a NEW_LINE DEDENT return gcd ( b , a % b ) NEW_LINE DEDENT
a = 98 NEW_LINE b = 56 NEW_LINE if ( gcd ( a , b ) ) : NEW_LINE INDENT print ( ' GCD ▁ of ' , a , ' and ' , b , ' is ' , gcd ( a , b ) ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( ' not ▁ found ' ) NEW_LINE DEDENT
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def reverseLevelOrder ( root ) : NEW_LINE INDENT h = height ( root ) NEW_LINE for i in reversed ( range ( 1 , h + 1 ) ) : NEW_LINE INDENT printGivenLevel ( root , i ) NEW_LINE DEDENT DEDENT
def printGivenLevel ( root , level ) : NEW_LINE INDENT if root is None : NEW_LINE INDENT return NEW_LINE DEDENT if level == 1 : NEW_LINE INDENT print root . data , NEW_LINE DEDENT elif level > 1 : NEW_LINE INDENT printGivenLevel ( root . left , level - 1 ) NEW_LINE printGivenLevel ( root . right , level - 1 ) NEW_LINE DEDENT DEDENT
def height ( node ) : NEW_LINE INDENT if node is None : NEW_LINE INDENT return 0 NEW_LINE DEDENT else : NEW_LINE DEDENT
lheight = height ( node . left ) NEW_LINE rheight = height ( node . right ) NEW_LINE
if lheight > rheight : NEW_LINE INDENT return lheight + 1 NEW_LINE DEDENT else : NEW_LINE INDENT return rheight + 1 NEW_LINE DEDENT
root = Node ( 1 ) NEW_LINE root . left = Node ( 2 ) NEW_LINE root . right = Node ( 3 ) NEW_LINE root . left . left = Node ( 4 ) NEW_LINE root . left . right = Node ( 5 ) NEW_LINE print " Level ▁ Order ▁ traversal ▁ of ▁ binary ▁ tree ▁ is " NEW_LINE reverseLevelOrder ( root ) NEW_LINE
def indexedSequentialSearch ( arr , n , k ) : NEW_LINE INDENT elements = [ 0 ] * 20 NEW_LINE indices = [ 0 ] * 20 NEW_LINE j , ind , start , end = 0 , 0 , 0 , 0 NEW_LINE set_flag = 0 NEW_LINE for i in range ( 0 , n , 3 ) : NEW_LINE DEDENT
elements [ ind ] = arr [ i ] NEW_LINE
indices [ ind ] = i NEW_LINE ind += 1 NEW_LINE if k < elements [ 0 ] : NEW_LINE print ( " Not ▁ found " ) NEW_LINE exit ( 0 ) NEW_LINE else : NEW_LINE for i in range ( 1 , ind + 1 ) : NEW_LINE INDENT if k <= elements [ i ] : NEW_LINE INDENT start = indices [ i - 1 ] NEW_LINE end = indices [ i ] NEW_LINE set_flag = 1 NEW_LINE break NEW_LINE DEDENT DEDENT if set_flag == 0 : NEW_LINE start = indices [ i - 1 ] NEW_LINE end = n NEW_LINE for i in range ( start , end + 1 ) : NEW_LINE if k == arr [ i ] : NEW_LINE INDENT j = 1 NEW_LINE break NEW_LINE DEDENT if j == 1 : NEW_LINE print ( " Found ▁ at ▁ index " , i ) NEW_LINE else : NEW_LINE print ( " Not ▁ found " ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 6 , 7 , 8 , 9 , 10 ] NEW_LINE n = len ( arr ) NEW_LINE DEDENT
k = 8 NEW_LINE
indexedSequentialSearch ( arr , n , k ) NEW_LINE
def printNSE ( arr ) : NEW_LINE INDENT for i in range ( 0 , len ( arr ) , 1 ) : NEW_LINE INDENT next = - 1 NEW_LINE for j in range ( i + 1 , len ( arr ) , 1 ) : NEW_LINE INDENT if arr [ i ] > arr [ j ] : NEW_LINE INDENT next = arr [ j ] NEW_LINE break NEW_LINE DEDENT DEDENT print ( str ( arr [ i ] ) + " ▁ - - ▁ " + str ( next ) ) NEW_LINE DEDENT DEDENT
arr = [ 11 , 13 , 21 , 3 ] NEW_LINE printNSE ( arr ) NEW_LINE
def bpm ( table , u , seen , matchR ) : NEW_LINE INDENT global M , N NEW_LINE DEDENT
for v in range ( N ) : NEW_LINE
if ( table [ u ] [ v ] > 0 and not seen [ v ] ) : NEW_LINE
seen [ v ] = True NEW_LINE
if ( matchR [ v ] < 0 or bpm ( table , matchR [ v ] , seen , matchR ) ) : NEW_LINE INDENT matchR [ v ] = u NEW_LINE return True NEW_LINE DEDENT return False NEW_LINE
def maxBPM ( table ) : NEW_LINE INDENT global M , N NEW_LINE DEDENT
matchR = [ - 1 ] * N NEW_LINE
result = 0 NEW_LINE for u in range ( M ) : NEW_LINE
seen = [ 0 ] * N NEW_LINE
if ( bpm ( table , u , seen , matchR ) ) : NEW_LINE INDENT result += 1 NEW_LINE DEDENT print ( " The ▁ number ▁ of ▁ maximum ▁ packets ▁ sent " , " in ▁ the ▁ time ▁ slot ▁ is " , result ) NEW_LINE for x in range ( N ) : NEW_LINE if ( matchR [ x ] + 1 != 0 ) : NEW_LINE INDENT print ( " T " , matchR [ x ] + 1 , " - > ▁ R " , x + 1 ) NEW_LINE DEDENT return result NEW_LINE
M = 3 NEW_LINE N = 4 NEW_LINE table = [ [ 0 , 2 , 0 ] , [ 3 , 0 , 1 ] , [ 2 , 4 , 0 ] ] NEW_LINE max_flow = maxBPM ( table ) NEW_LINE
NO_OF_CHARS = 256 NEW_LINE
def printList ( list , word , list_size ) : NEW_LINE INDENT map = [ 0 ] * NO_OF_CHARS NEW_LINE DEDENT
for i in word : NEW_LINE INDENT map [ ord ( i ) ] = 1 NEW_LINE DEDENT
word_size = len ( word ) NEW_LINE
for i in list : NEW_LINE INDENT count = 0 NEW_LINE for j in i : NEW_LINE INDENT if map [ ord ( j ) ] : NEW_LINE INDENT count += 1 NEW_LINE DEDENT DEDENT DEDENT
map [ ord ( j ) ] = 0 NEW_LINE if count == word_size : NEW_LINE print i NEW_LINE
for j in xrange ( len ( word ) ) : NEW_LINE INDENT map [ ord ( word [ j ] ) ] = 1 NEW_LINE DEDENT
string = " sun " NEW_LINE list = [ " geeksforgeeks " , " unsorted " , " sunday " , " just " , " sss " ] NEW_LINE printList ( list , string , 5 ) NEW_LINE
NO_OF_CHARS = 256 NEW_LINE
def getCharCountArray ( string ) : NEW_LINE INDENT count = [ 0 ] * NO_OF_CHARS NEW_LINE for i in string : NEW_LINE INDENT count [ ord ( i ) ] += 1 NEW_LINE DEDENT return count NEW_LINE DEDENT
def firstNonRepeating ( string ) : NEW_LINE INDENT count = getCharCountArray ( string ) NEW_LINE index = - 1 NEW_LINE k = 0 NEW_LINE for i in string : NEW_LINE INDENT if count [ ord ( i ) ] == 1 : NEW_LINE INDENT index = k NEW_LINE break NEW_LINE DEDENT k += 1 NEW_LINE DEDENT return index NEW_LINE DEDENT
string = " geeksforgeeks " NEW_LINE index = firstNonRepeating ( string ) NEW_LINE if index == 1 : NEW_LINE INDENT print " Either ▁ all ▁ characters ▁ are ▁ repeating ▁ or ▁ string ▁ is ▁ empty " NEW_LINE DEDENT else : NEW_LINE INDENT print " First ▁ non - repeating ▁ character ▁ is ▁ " + string [ index ] NEW_LINE DEDENT
def divideString ( string , n ) : NEW_LINE INDENT str_size = len ( string ) NEW_LINE DEDENT
if str_size % n != 0 : NEW_LINE INDENT print " Invalid ▁ Input : ▁ String ▁ size ▁ is ▁ not ▁ divisible ▁ by ▁ n " NEW_LINE return NEW_LINE DEDENT
part_size = str_size / n NEW_LINE k = 0 NEW_LINE for i in string : NEW_LINE INDENT if k % part_size == 0 : NEW_LINE INDENT print   " NEW_LINE DEDENT DEDENT " , NEW_LINE INDENT print i , NEW_LINE k += 1 NEW_LINE DEDENT
/ * length od string is 28 * / NEW_LINE
divideString ( string , 4 ) NEW_LINE
def collinear ( x1 , y1 , x2 , y2 , x3 , y3 ) : NEW_LINE INDENT if ( ( y3 - y2 ) * ( x2 - x1 ) == ( y2 - y1 ) * ( x3 - x2 ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT DEDENT
x1 , x2 , x3 , y1 , y2 , y3 = 1 , 1 , 0 , 1 , 6 , 9 NEW_LINE collinear ( x1 , y1 , x2 , y2 , x3 , y3 ) ; NEW_LINE
def bestApproximate ( x , y , n ) : NEW_LINE INDENT sum_x = 0 NEW_LINE sum_y = 0 NEW_LINE sum_xy = 0 NEW_LINE sum_x2 = 0 NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT sum_x += x [ i ] NEW_LINE sum_y += y [ i ] NEW_LINE sum_xy += x [ i ] * y [ i ] NEW_LINE sum_x2 += pow ( x [ i ] , 2 ) NEW_LINE DEDENT m = ( float ) ( ( n * sum_xy - sum_x * sum_y ) / ( n * sum_x2 - pow ( sum_x , 2 ) ) ) ; NEW_LINE c = ( float ) ( sum_y - m * sum_x ) / n ; NEW_LINE print ( " m ▁ = ▁ " , m ) ; NEW_LINE print ( " c ▁ = ▁ " , c ) ; NEW_LINE DEDENT
x = [ 1 , 2 , 3 , 4 , 5 ] NEW_LINE y = [ 14 , 27 , 40 , 55 , 68 ] NEW_LINE n = len ( x ) NEW_LINE bestApproximate ( x , y , n ) NEW_LINE
import sys NEW_LINE
def findMinInsertions ( str , l , h ) : NEW_LINE
if ( l > h ) : NEW_LINE INDENT return sys . maxsize NEW_LINE DEDENT if ( l == h ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT if ( l == h - 1 ) : NEW_LINE INDENT return 0 if ( str [ l ] == str [ h ] ) else 1 NEW_LINE DEDENT
if ( str [ l ] == str [ h ] ) : NEW_LINE INDENT return findMinInsertions ( str , l + 1 , h - 1 ) NEW_LINE DEDENT else : NEW_LINE INDENT return ( min ( findMinInsertions ( str , l , h - 1 ) , findMinInsertions ( str , l + 1 , h ) ) + 1 ) NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT str = " geeks " NEW_LINE print ( findMinInsertions ( str , 0 , len ( str ) - 1 ) ) NEW_LINE DEDENT
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def MorrisTraversal ( root ) : NEW_LINE INDENT curr = root NEW_LINE while curr : NEW_LINE DEDENT
if curr . left is None : NEW_LINE INDENT print ( curr . data , end = " ▁ " ) NEW_LINE curr = curr . right NEW_LINE DEDENT else : NEW_LINE
prev = curr . left NEW_LINE while prev . right is not None and prev . right is not curr : NEW_LINE INDENT prev = prev . right NEW_LINE DEDENT
if prev . right is curr : NEW_LINE INDENT prev . right = None NEW_LINE curr = curr . right NEW_LINE DEDENT
else : NEW_LINE INDENT print ( curr . data , end = " ▁ " ) NEW_LINE prev . right = curr NEW_LINE curr = curr . left NEW_LINE DEDENT
def preorfer ( root ) : NEW_LINE INDENT if root : NEW_LINE INDENT print ( root . data , end = " ▁ " ) NEW_LINE preorfer ( root . left ) NEW_LINE preorfer ( root . right ) NEW_LINE DEDENT DEDENT
root = Node ( 1 ) NEW_LINE root . left = Node ( 2 ) NEW_LINE root . right = Node ( 3 ) NEW_LINE root . left . left = Node ( 4 ) NEW_LINE root . left . right = Node ( 5 ) NEW_LINE root . right . left = Node ( 6 ) NEW_LINE root . right . right = Node ( 7 ) NEW_LINE root . left . left . left = Node ( 8 ) NEW_LINE root . left . left . right = Node ( 9 ) NEW_LINE root . left . right . left = Node ( 10 ) NEW_LINE root . left . right . right = Node ( 11 ) NEW_LINE MorrisTraversal ( root ) NEW_LINE print ( " " ) NEW_LINE preorfer ( root ) NEW_LINE
def push ( self , new_data ) : NEW_LINE
new_node = Node ( new_data ) NEW_LINE
new_node . next = self . head NEW_LINE
self . head = new_node NEW_LINE
def insertAfter ( self , prev_node , new_data ) : NEW_LINE
if prev_node is None : NEW_LINE INDENT print " The ▁ given ▁ previous ▁ node ▁ must ▁ inLinkedList . " NEW_LINE return NEW_LINE DEDENT
new_node = Node ( new_data ) NEW_LINE
new_node . next = prev_node . next NEW_LINE
prev_node . next = new_node NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . next = None NEW_LINE DEDENT DEDENT class LinkedList : NEW_LINE
def isPalindrome ( self , head ) : NEW_LINE INDENT slow_ptr = head NEW_LINE fast_ptr = head NEW_LINE prev_of_slow_ptr = head NEW_LINE DEDENT
midnode = None NEW_LINE
res = True NEW_LINE if ( head != None and head . next != None ) : NEW_LINE
while ( fast_ptr != None and fast_ptr . next != None ) : NEW_LINE
fast_ptr = fast_ptr . next . next NEW_LINE prev_of_slow_ptr = slow_ptr NEW_LINE slow_ptr = slow_ptr . next NEW_LINE
if ( fast_ptr != None ) : NEW_LINE INDENT midnode = slow_ptr NEW_LINE slow_ptr = slow_ptr . next NEW_LINE DEDENT
second_half = slow_ptr NEW_LINE
prev_of_slow_ptr . next = None NEW_LINE
second_half = self . reverse ( second_half ) NEW_LINE
res = self . compareLists ( head , second_half ) NEW_LINE
second_half = self . reverse ( second_half ) NEW_LINE if ( midnode != None ) : NEW_LINE
prev_of_slow_ptr . next = midnode NEW_LINE midnode . next = second_half NEW_LINE else : NEW_LINE prev_of_slow_ptr . next = second_half NEW_LINE return res NEW_LINE
def reverse ( self , second_half ) : NEW_LINE INDENT prev = None NEW_LINE current = second_half NEW_LINE next = None NEW_LINE while current != None : NEW_LINE INDENT next = current . next NEW_LINE current . next = prev NEW_LINE prev = current NEW_LINE current = next NEW_LINE DEDENT second_half = prev NEW_LINE return second_half NEW_LINE DEDENT
def compareLists ( self , head1 , head2 ) : NEW_LINE INDENT temp1 = head1 NEW_LINE temp2 = head2 NEW_LINE while ( temp1 and temp2 ) : NEW_LINE INDENT if ( temp1 . data == temp2 . data ) : NEW_LINE INDENT temp1 = temp1 . next NEW_LINE temp2 = temp2 . next NEW_LINE DEDENT else : NEW_LINE INDENT return 0 NEW_LINE DEDENT DEDENT DEDENT
if ( temp1 == None and temp2 == None ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT
return 0 NEW_LINE
def push ( self , new_data ) : NEW_LINE
new_node = Node ( new_data ) NEW_LINE
new_node . next = self . head NEW_LINE
self . head = new_node NEW_LINE
def printList ( self ) : NEW_LINE INDENT temp = self . head NEW_LINE while ( temp ) : NEW_LINE INDENT print ( temp . data , end = " - > " ) NEW_LINE temp = temp . next NEW_LINE DEDENT print ( " NULL " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE
l = LinkedList ( ) NEW_LINE s = [ ' a ' , ' b ' , ' a ' , ' c ' , ' a ' , ' b ' , ' a ' ] NEW_LINE for i in range ( 7 ) : NEW_LINE INDENT l . push ( s [ i ] ) NEW_LINE l . printList ( ) NEW_LINE if ( l . isPalindrome ( l . head ) != False ) : NEW_LINE INDENT print ( " Is Palindrome " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Not Palindrome " ) NEW_LINE DEDENT print ( ) NEW_LINE DEDENT
class LinkedList ( object ) : NEW_LINE INDENT def __init__ ( self ) : NEW_LINE INDENT self . head = None NEW_LINE DEDENT DEDENT
def swapNodes ( self , x , y ) : NEW_LINE
if x == y : NEW_LINE INDENT return NEW_LINE DEDENT
prevX = None NEW_LINE currX = self . head NEW_LINE while currX != None and currX . data != x : NEW_LINE INDENT prevX = currX NEW_LINE currX = currX . next NEW_LINE DEDENT
prevY = None NEW_LINE currY = self . head NEW_LINE while currY != None and currY . data != y : NEW_LINE INDENT prevY = currY NEW_LINE currY = currY . next NEW_LINE DEDENT
if currX == None or currY == None : NEW_LINE INDENT return NEW_LINE DEDENT
if prevX != None : NEW_LINE INDENT prevX . next = currY NEW_LINE DEDENT
else : NEW_LINE INDENT self . head = currY NEW_LINE DEDENT
if prevY != None : NEW_LINE INDENT prevY . next = currX NEW_LINE DEDENT
else : NEW_LINE INDENT self . head = currX NEW_LINE DEDENT
temp = currX . next NEW_LINE currX . next = currY . next NEW_LINE currY . next = temp NEW_LINE
def push ( self , new_data ) : NEW_LINE
new_Node = self . Node ( new_data ) NEW_LINE
new_Node . next = self . head NEW_LINE
self . head = new_Node NEW_LINE
def printList ( self ) : NEW_LINE INDENT tNode = self . head NEW_LINE while tNode != None : NEW_LINE INDENT print tNode . data , NEW_LINE tNode = tNode . next NEW_LINE DEDENT DEDENT
llist = LinkedList ( ) NEW_LINE
llist . push ( 7 ) NEW_LINE llist . push ( 6 ) NEW_LINE llist . push ( 5 ) NEW_LINE llist . push ( 4 ) NEW_LINE llist . push ( 3 ) NEW_LINE llist . push ( 2 ) NEW_LINE llist . push ( 1 ) NEW_LINE print " Linked ▁ list ▁ before ▁ calling ▁ swapNodes ( ) ▁ " NEW_LINE llist . printList ( ) NEW_LINE llist . swapNodes ( 4 , 3 ) NEW_LINE print   " NEW_LINE Linked list after calling swapNodes ( )   " NEW_LINE llist . printList ( ) NEW_LINE
def push ( self , new_data ) : NEW_LINE
new_node = Node ( data = new_data ) NEW_LINE
new_node . next = self . head NEW_LINE new_node . prev = None NEW_LINE
if self . head is not None : NEW_LINE INDENT self . head . prev = new_node NEW_LINE DEDENT
self . head = new_node NEW_LINE
def insertAfter ( self , prev_node , new_data ) : NEW_LINE
if prev_node is None : NEW_LINE INDENT print ( " This ▁ node ▁ doesn ' t ▁ exist ▁ in ▁ DLL " ) NEW_LINE return NEW_LINE DEDENT
new_node = Node ( data = new_data ) NEW_LINE
new_node . next = prev_node . next NEW_LINE
prev_node . next = new_node NEW_LINE
new_node . prev = prev_node NEW_LINE
if new_node . next is not None : NEW_LINE INDENT new_node . next . prev = new_node NEW_LINE DEDENT
def append ( self , new_data ) : NEW_LINE
new_node = Node ( data = new_data ) NEW_LINE last = self . head NEW_LINE
new_node . next = None NEW_LINE
if self . head is None : NEW_LINE INDENT new_node . prev = None NEW_LINE self . head = new_node NEW_LINE return NEW_LINE DEDENT
while ( last . next is not None ) : NEW_LINE INDENT last = last . next NEW_LINE DEDENT
last . next = new_node NEW_LINE
new_node . prev = last NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . info = data NEW_LINE self . next = None NEW_LINE self . prev = None NEW_LINE DEDENT DEDENT head = None NEW_LINE tail = None NEW_LINE
def nodeInsetail ( key ) : NEW_LINE INDENT global head NEW_LINE global tail NEW_LINE p = Node ( 0 ) NEW_LINE p . info = key NEW_LINE p . next = None NEW_LINE DEDENT
if ( ( head ) == None ) : NEW_LINE INDENT ( head ) = p NEW_LINE ( tail ) = p NEW_LINE ( head ) . prev = None NEW_LINE return NEW_LINE DEDENT
if ( ( p . info ) < ( ( head ) . info ) ) : NEW_LINE INDENT p . prev = None NEW_LINE ( head ) . prev = p NEW_LINE p . next = ( head ) NEW_LINE ( head ) = p NEW_LINE return NEW_LINE DEDENT
if ( ( p . info ) > ( ( tail ) . info ) ) : NEW_LINE INDENT p . prev = ( tail ) NEW_LINE ( tail ) . next = p NEW_LINE ( tail ) = p NEW_LINE return NEW_LINE DEDENT
temp = ( head ) . next NEW_LINE while ( ( temp . info ) < ( p . info ) ) : NEW_LINE INDENT temp = temp . next NEW_LINE DEDENT
( temp . prev ) . next = p NEW_LINE p . prev = temp . prev NEW_LINE temp . prev = p NEW_LINE p . next = temp NEW_LINE
def printList ( temp ) : NEW_LINE INDENT while ( temp != None ) : NEW_LINE INDENT print ( temp . info , end = " ▁ " ) NEW_LINE temp = temp . next NEW_LINE DEDENT DEDENT
nodeInsetail ( 30 ) NEW_LINE nodeInsetail ( 50 ) NEW_LINE nodeInsetail ( 90 ) NEW_LINE nodeInsetail ( 10 ) NEW_LINE nodeInsetail ( 40 ) NEW_LINE nodeInsetail ( 110 ) NEW_LINE nodeInsetail ( 60 ) NEW_LINE nodeInsetail ( 95 ) NEW_LINE nodeInsetail ( 23 ) NEW_LINE print ( " Doubly linked list on printing from left to right " ) NEW_LINE printList ( head ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . next = None NEW_LINE DEDENT DEDENT
def fun1 ( head ) : NEW_LINE INDENT if ( head == None ) : NEW_LINE INDENT return NEW_LINE DEDENT fun1 ( head . next ) NEW_LINE print ( head . data , end = " ▁ " ) NEW_LINE DEDENT
def fun2 ( head ) : NEW_LINE INDENT if ( head == None ) : NEW_LINE INDENT return NEW_LINE DEDENT print ( head . data , end = " ▁ " ) NEW_LINE if ( head . next != None ) : NEW_LINE INDENT fun2 ( head . next . next ) NEW_LINE DEDENT print ( head . data , end = " ▁ " ) NEW_LINE DEDENT
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . next = None NEW_LINE DEDENT DEDENT
def fun1 ( head ) : NEW_LINE INDENT if ( head == None ) : NEW_LINE INDENT return NEW_LINE DEDENT fun1 ( head . next ) NEW_LINE print ( head . data , end = " ▁ " ) NEW_LINE DEDENT
def fun2 ( start ) : NEW_LINE INDENT if ( start == None ) : NEW_LINE INDENT return NEW_LINE DEDENT print ( start . data , end = " ▁ " ) NEW_LINE if ( start . next != None ) : NEW_LINE INDENT fun2 ( start . next . next ) NEW_LINE DEDENT print ( start . data , end = " ▁ " ) NEW_LINE DEDENT
def push ( head , new_data ) : NEW_LINE
new_node = Node ( new_data ) NEW_LINE
new_node . next = head NEW_LINE
head = new_node NEW_LINE return head NEW_LINE
head = None NEW_LINE
head = Node ( 5 ) NEW_LINE head = push ( head , 4 ) NEW_LINE head = push ( head , 3 ) NEW_LINE head = push ( head , 2 ) NEW_LINE head = push ( head , 1 ) NEW_LINE print ( " Output ▁ of ▁ fun1 ( ) ▁ for ▁ list ▁ 1 - > 2 - > 3 - > 4 - > 5" ) NEW_LINE fun1 ( head ) NEW_LINE print ( " Output of fun2 ( ) for list 1 -> 2 -> 3 -> 4 -> 5 " ) NEW_LINE fun2 ( head ) NEW_LINE
class Node : NEW_LINE
def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . next = None NEW_LINE DEDENT
def printsqrtn ( head ) : NEW_LINE INDENT sqrtn = None NEW_LINE i = 1 NEW_LINE j = 1 NEW_LINE DEDENT
while ( head != None ) : NEW_LINE
if ( i == j * j ) : NEW_LINE
if ( sqrtn == None ) : NEW_LINE INDENT sqrtn = head NEW_LINE DEDENT else : NEW_LINE INDENT sqrtn = sqrtn . next NEW_LINE DEDENT
j = j + 1 NEW_LINE i = i + 1 NEW_LINE head = head . next NEW_LINE
return sqrtn . data NEW_LINE def print_1 ( head ) : NEW_LINE while ( head != None ) : NEW_LINE INDENT print ( head . data , end = " ▁ " ) NEW_LINE head = head . next NEW_LINE DEDENT print ( " ▁ " ) NEW_LINE
def push ( head_ref , new_data ) : NEW_LINE
new_node = Node ( 0 ) NEW_LINE
new_node . data = new_data NEW_LINE
new_node . next = ( head_ref ) NEW_LINE
( head_ref ) = new_node NEW_LINE return head_ref NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
head = None NEW_LINE head = push ( head , 40 ) NEW_LINE head = push ( head , 30 ) NEW_LINE head = push ( head , 20 ) NEW_LINE head = push ( head , 10 ) NEW_LINE print ( " Given ▁ linked ▁ list ▁ is : " ) NEW_LINE print_1 ( head ) NEW_LINE print ( " sqrt ( n ) th ▁ node ▁ is ▁ " , printsqrtn ( head ) ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , key ) : NEW_LINE INDENT self . data = key NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def insert ( node , data ) : NEW_LINE
if node is None : NEW_LINE INDENT return ( Node ( data ) ) NEW_LINE DEDENT else : NEW_LINE
if data <= node . data : NEW_LINE INDENT node . left = insert ( node . left , data ) NEW_LINE DEDENT else : NEW_LINE INDENT node . right = insert ( node . right , data ) NEW_LINE DEDENT
return node NEW_LINE
def minValue ( node ) : NEW_LINE INDENT current = node NEW_LINE DEDENT
while ( current . left is not None ) : NEW_LINE INDENT current = current . left NEW_LINE DEDENT return current . data NEW_LINE
root = None NEW_LINE root = insert ( root , 4 ) NEW_LINE insert ( root , 2 ) NEW_LINE insert ( root , 1 ) NEW_LINE insert ( root , 3 ) NEW_LINE insert ( root , 6 ) NEW_LINE insert ( root , 5 ) NEW_LINE print   " NEW_LINE Minimum value in BST is % d " % ( minValue ( root ) ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def buildTree ( inOrder , preOrder , inStrt , inEnd ) : NEW_LINE INDENT if ( inStrt > inEnd ) : NEW_LINE INDENT return None NEW_LINE DEDENT DEDENT
tNode = Node ( preOrder [ buildTree . preIndex ] ) NEW_LINE buildTree . preIndex += 1 NEW_LINE
if inStrt == inEnd : NEW_LINE INDENT return tNode NEW_LINE DEDENT
inIndex = search ( inOrder , inStrt , inEnd , tNode . data ) NEW_LINE
tNode . left = buildTree ( inOrder , preOrder , inStrt , inIndex - 1 ) NEW_LINE tNode . right = buildTree ( inOrder , preOrder , inIndex + 1 , inEnd ) NEW_LINE return tNode NEW_LINE
def search ( arr , start , end , value ) : NEW_LINE INDENT for i in range ( start , end + 1 ) : NEW_LINE INDENT if arr [ i ] == value : NEW_LINE INDENT return i NEW_LINE DEDENT DEDENT DEDENT
def printInorder ( node ) : NEW_LINE INDENT if node is None : NEW_LINE INDENT return NEW_LINE DEDENT DEDENT
printInorder ( node . left ) NEW_LINE
print node . data , NEW_LINE
printInorder ( node . right ) NEW_LINE
inOrder = [ ' D ' , ' B ' , ' E ' , ' A ' , ' F ' , ' C ' ] NEW_LINE preOrder = [ ' A ' , ' B ' , ' D ' , ' E ' , ' C ' , ' F ' ] NEW_LINE buildTree . preIndex = 0 NEW_LINE root = buildTree ( inOrder , preOrder , 0 , len ( inOrder ) - 1 ) NEW_LINE
print " Inorder ▁ traversal ▁ of ▁ the ▁ constructed ▁ tree ▁ is " NEW_LINE printInorder ( root ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def lca ( root , n1 , n2 ) : NEW_LINE INDENT while root : NEW_LINE DEDENT
if root . data > n1 and root . data > n2 : NEW_LINE INDENT root = root . left NEW_LINE DEDENT
elif root . data < n1 and root . data < n2 : NEW_LINE INDENT root = root . right NEW_LINE DEDENT else : NEW_LINE INDENT break NEW_LINE DEDENT return root NEW_LINE
root = Node ( 20 ) NEW_LINE root . left = Node ( 8 ) NEW_LINE root . right = Node ( 22 ) NEW_LINE root . left . left = Node ( 4 ) NEW_LINE root . left . right = Node ( 12 ) NEW_LINE root . left . right . left = Node ( 10 ) NEW_LINE root . left . right . right = Node ( 14 ) NEW_LINE n1 = 10 ; n2 = 14 NEW_LINE t = lca ( root , n1 , n2 ) NEW_LINE print " LCA ▁ of ▁ % d ▁ and ▁ % d ▁ is ▁ % d " % ( n1 , n2 , t . data ) NEW_LINE n1 = 14 ; n2 = 8 NEW_LINE t = lca ( root , n1 , n2 ) NEW_LINE print " LCA ▁ of ▁ % d ▁ and ▁ % d ▁ is ▁ % d " % ( n1 , n2 , t . data ) NEW_LINE n1 = 10 ; n2 = 22 NEW_LINE t = lca ( root , n1 , n2 ) NEW_LINE print " LCA ▁ of ▁ % d ▁ and ▁ % d ▁ is ▁ % d " % ( n1 , n2 , t . data ) NEW_LINE
def hasOnlyOneChild ( pre , size ) : NEW_LINE INDENT nextDiff = 0 ; lastDiff = 0 NEW_LINE for i in range ( size - 1 ) : NEW_LINE INDENT nextDiff = pre [ i ] - pre [ i + 1 ] NEW_LINE lastDiff = pre [ i ] - pre [ size - 1 ] NEW_LINE if nextDiff * lastDiff < 0 : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT return True NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT pre = [ 8 , 3 , 5 , 7 , 6 ] NEW_LINE size = len ( pre ) NEW_LINE if ( hasOnlyOneChild ( pre , size ) == True ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT DEDENT
def hasOnlyOneChild ( pre , size ) : NEW_LINE
min = 0 ; max = 0 NEW_LINE if pre [ size - 1 ] > pre [ size - 2 ] : NEW_LINE INDENT max = pre [ size - 1 ] NEW_LINE min = pre [ size - 2 ] NEW_LINE DEDENT else : NEW_LINE INDENT max = pre [ size - 2 ] NEW_LINE min = pre [ size - 1 ] NEW_LINE DEDENT
for i in range ( size - 3 , 0 , - 1 ) : NEW_LINE INDENT if pre [ i ] < min : NEW_LINE INDENT min = pre [ i ] NEW_LINE DEDENT elif pre [ i ] > max : NEW_LINE INDENT max = pre [ i ] NEW_LINE DEDENT else : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT return True NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT pre = [ 8 , 3 , 5 , 7 , 6 ] NEW_LINE size = len ( pre ) NEW_LINE if ( hasOnlyOneChild ( pre , size ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT DEDENT
class Node : NEW_LINE INDENT def __init__ ( self , key ) : NEW_LINE INDENT self . data = key NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def insert ( node , data ) : NEW_LINE
if node is None : NEW_LINE INDENT return Node ( data ) NEW_LINE DEDENT else : NEW_LINE
if data <= node . data : NEW_LINE INDENT temp = insert ( node . left , data ) NEW_LINE node . left = temp NEW_LINE temp . parent = node NEW_LINE DEDENT else : NEW_LINE INDENT temp = insert ( node . right , data ) NEW_LINE node . right = temp NEW_LINE temp . parent = node NEW_LINE DEDENT
return node NEW_LINE def inOrderSuccessor ( n ) : NEW_LINE
if n . right is not None : NEW_LINE INDENT return minValue ( n . right ) NEW_LINE DEDENT
p = n . parent NEW_LINE while ( p is not None ) : NEW_LINE INDENT if n != p . right : NEW_LINE INDENT break NEW_LINE DEDENT n = p NEW_LINE p = p . parent NEW_LINE DEDENT return p NEW_LINE
def minValue ( node ) : NEW_LINE INDENT current = node NEW_LINE DEDENT
while ( current is not None ) : NEW_LINE INDENT if current . left is None : NEW_LINE INDENT break NEW_LINE DEDENT current = current . left NEW_LINE DEDENT return current NEW_LINE
root = None NEW_LINE root = insert ( root , 20 ) NEW_LINE root = insert ( root , 8 ) ; NEW_LINE root = insert ( root , 22 ) ; NEW_LINE root = insert ( root , 4 ) ; NEW_LINE root = insert ( root , 12 ) ; NEW_LINE root = insert ( root , 10 ) ; NEW_LINE root = insert ( root , 14 ) ; NEW_LINE temp = root . left . right . right NEW_LINE succ = inOrderSuccessor ( root , temp ) NEW_LINE if succ is not None : NEW_LINE INDENT print " \n Inorder ▁ Successor ▁ of ▁ % ▁ d ▁ is ▁ % ▁ d ▁ " % ( temp . data , succ . data ) NEW_LINE DEDENT else : NEW_LINE INDENT print " \n Inorder ▁ Successor ▁ doesn ' t ▁ exist " NEW_LINE DEDENT
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def constructTreeUtil ( pre : list , post : list , l : int , h : int , size : int ) -> Node : NEW_LINE INDENT global preIndex NEW_LINE DEDENT
if ( preIndex >= size or l > h ) : NEW_LINE INDENT return None NEW_LINE DEDENT
root = Node ( pre [ preIndex ] ) NEW_LINE preIndex += 1 NEW_LINE
if ( l == h or preIndex >= size ) : NEW_LINE INDENT return root NEW_LINE DEDENT
i = l NEW_LINE while i <= h : NEW_LINE INDENT if ( pre [ preIndex ] == post [ i ] ) : NEW_LINE INDENT break NEW_LINE DEDENT i += 1 NEW_LINE DEDENT
if ( i <= h ) : NEW_LINE INDENT root . left = constructTreeUtil ( pre , post , l , i , size ) NEW_LINE root . right = constructTreeUtil ( pre , post , i + 1 , h , size ) NEW_LINE DEDENT return root NEW_LINE
def constructTree ( pre : list , post : list , size : int ) -> Node : NEW_LINE INDENT global preIndex NEW_LINE return constructTreeUtil ( pre , post , 0 , size - 1 , size ) NEW_LINE DEDENT
def printInorder ( node : Node ) -> None : NEW_LINE INDENT if ( node is None ) : NEW_LINE INDENT return NEW_LINE DEDENT printInorder ( node . left ) NEW_LINE print ( node . data , end = " ▁ " ) NEW_LINE printInorder ( node . right ) NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT pre = [ 1 , 2 , 4 , 8 , 9 , 5 , 3 , 6 , 7 ] NEW_LINE post = [ 8 , 9 , 4 , 5 , 2 , 6 , 7 , 3 , 1 ] NEW_LINE size = len ( pre ) NEW_LINE preIndex = 0 NEW_LINE root = constructTree ( pre , post , size ) NEW_LINE print ( " Inorder ▁ traversal ▁ of ▁ " " the ▁ constructed ▁ tree : ▁ " ) NEW_LINE printInorder ( root ) NEW_LINE DEDENT
def printSorted ( arr , start , end ) : NEW_LINE INDENT if start > end : NEW_LINE INDENT return NEW_LINE DEDENT DEDENT
printSorted ( arr , start * 2 + 1 , end ) NEW_LINE
print ( arr [ start ] , end = " ▁ " ) NEW_LINE
printSorted ( arr , start * 2 + 2 , end ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 4 , 2 , 5 , 1 , 3 ] NEW_LINE arr_size = len ( arr ) NEW_LINE printSorted ( arr , 0 , arr_size - 1 ) NEW_LINE DEDENT
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . key = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def ceil ( root , inp ) : NEW_LINE
if root == None : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
if root . key == inp : NEW_LINE INDENT return root . key NEW_LINE DEDENT
if root . key < inp : NEW_LINE INDENT return ceil ( root . right , inp ) NEW_LINE DEDENT
val = ceil ( root . left , inp ) NEW_LINE return val if val >= inp else root . key NEW_LINE
root = Node ( 8 ) NEW_LINE root . left = Node ( 4 ) NEW_LINE root . right = Node ( 12 ) NEW_LINE root . left . left = Node ( 2 ) NEW_LINE root . left . right = Node ( 6 ) NEW_LINE root . right . left = Node ( 10 ) NEW_LINE root . right . right = Node ( 14 ) NEW_LINE for i in range ( 16 ) : NEW_LINE INDENT print " % ▁ d ▁ % ▁ d " % ( i , ceil ( root , i ) ) NEW_LINE DEDENT
class newNode : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . key = data NEW_LINE self . count = 1 NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def inorder ( root ) : NEW_LINE INDENT if root != None : NEW_LINE INDENT inorder ( root . left ) NEW_LINE print ( root . key , " ( " , root . count , " ) " , end = " ▁ " ) NEW_LINE inorder ( root . right ) NEW_LINE DEDENT DEDENT
def insert ( node , key ) : NEW_LINE
if node == None : NEW_LINE INDENT k = newNode ( key ) NEW_LINE return k NEW_LINE DEDENT
if key == node . key : NEW_LINE INDENT ( node . count ) += 1 NEW_LINE return node NEW_LINE DEDENT
if key < node . key : NEW_LINE INDENT node . left = insert ( node . left , key ) NEW_LINE DEDENT else : NEW_LINE INDENT node . right = insert ( node . right , key ) NEW_LINE DEDENT
return node NEW_LINE
def minValueNode ( node ) : NEW_LINE INDENT current = node NEW_LINE DEDENT
while current . left != None : NEW_LINE INDENT current = current . left NEW_LINE DEDENT return current NEW_LINE
def deleteNode ( root , key ) : NEW_LINE
if root == None : NEW_LINE INDENT return root NEW_LINE DEDENT
if key < root . key : NEW_LINE INDENT root . left = deleteNode ( root . left , key ) NEW_LINE DEDENT
elif key > root . key : NEW_LINE INDENT root . right = deleteNode ( root . right , key ) NEW_LINE DEDENT
else : NEW_LINE
if root . count > 1 : NEW_LINE INDENT root . count -= 1 NEW_LINE return root NEW_LINE DEDENT
if root . left == None : NEW_LINE INDENT temp = root . right NEW_LINE return temp NEW_LINE DEDENT elif root . right == None : NEW_LINE INDENT temp = root . left NEW_LINE return temp NEW_LINE DEDENT
temp = minValueNode ( root . right ) NEW_LINE
root . key = temp . key NEW_LINE
root . right = deleteNode ( root . right , temp . key ) NEW_LINE return root NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
root = None NEW_LINE root = insert ( root , 12 ) NEW_LINE root = insert ( root , 10 ) NEW_LINE root = insert ( root , 20 ) NEW_LINE root = insert ( root , 9 ) NEW_LINE root = insert ( root , 11 ) NEW_LINE root = insert ( root , 10 ) NEW_LINE root = insert ( root , 12 ) NEW_LINE root = insert ( root , 12 ) NEW_LINE print ( " Inorder ▁ traversal ▁ of ▁ the ▁ given ▁ tree " ) NEW_LINE inorder ( root ) NEW_LINE print ( ) NEW_LINE print ( " Delete ▁ 20" ) NEW_LINE root = deleteNode ( root , 20 ) NEW_LINE print ( " Inorder ▁ traversal ▁ of ▁ the ▁ modified ▁ tree " ) NEW_LINE inorder ( root ) NEW_LINE print ( ) NEW_LINE print ( " Delete ▁ 12" ) NEW_LINE root = deleteNode ( root , 12 ) NEW_LINE print ( " Inorder ▁ traversal ▁ of ▁ the ▁ modified ▁ tree " ) NEW_LINE inorder ( root ) NEW_LINE print ( ) NEW_LINE print ( " Delete ▁ 9" ) NEW_LINE root = deleteNode ( root , 9 ) NEW_LINE print ( " Inorder ▁ traversal ▁ of ▁ the ▁ modified ▁ tree " ) NEW_LINE inorder ( root ) NEW_LINE
class newNode : NEW_LINE INDENT def __init__ ( self , key ) : NEW_LINE INDENT self . key = key NEW_LINE self . left = self . right = None NEW_LINE DEDENT DEDENT
def inorder ( root ) : NEW_LINE INDENT if root != None : NEW_LINE INDENT inorder ( root . left ) NEW_LINE print ( root . key , end = " ▁ " ) NEW_LINE inorder ( root . right ) NEW_LINE DEDENT DEDENT
def insert ( node , key ) : NEW_LINE
if node == None : NEW_LINE INDENT return newNode ( key ) NEW_LINE DEDENT
if key < node . key : NEW_LINE INDENT node . left = insert ( node . left , key ) NEW_LINE DEDENT else : NEW_LINE INDENT node . right = insert ( node . right , key ) NEW_LINE DEDENT
return node NEW_LINE
def minValueNode ( node ) : NEW_LINE INDENT current = node NEW_LINE DEDENT
while current . left != None : NEW_LINE INDENT current = current . left NEW_LINE DEDENT return current NEW_LINE
def deleteNode ( root , key ) : NEW_LINE
if root == None : NEW_LINE INDENT return root NEW_LINE DEDENT
if key < root . key : NEW_LINE INDENT root . left = deleteNode ( root . left , key ) NEW_LINE DEDENT
elif key > root . key : NEW_LINE INDENT root . right = deleteNode ( root . right , key ) NEW_LINE DEDENT
else : NEW_LINE
if root . left == None : NEW_LINE INDENT temp = root . right NEW_LINE return temp NEW_LINE DEDENT elif root . right == None : NEW_LINE INDENT temp = root . left NEW_LINE return temp NEW_LINE DEDENT
temp = minValueNode ( root . right ) NEW_LINE
root . key = temp . key NEW_LINE
root . right = deleteNode ( root . right , temp . key ) NEW_LINE return root NEW_LINE
def changeKey ( root , oldVal , newVal ) : NEW_LINE
root = deleteNode ( root , oldVal ) NEW_LINE
root = insert ( root , newVal ) NEW_LINE
return root NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
root = None NEW_LINE root = insert ( root , 50 ) NEW_LINE root = insert ( root , 30 ) NEW_LINE root = insert ( root , 20 ) NEW_LINE root = insert ( root , 40 ) NEW_LINE root = insert ( root , 70 ) NEW_LINE root = insert ( root , 60 ) NEW_LINE root = insert ( root , 80 ) NEW_LINE print ( " Inorder ▁ traversal ▁ of ▁ the ▁ given ▁ tree " ) NEW_LINE inorder ( root ) NEW_LINE root = changeKey ( root , 40 , 10 ) NEW_LINE print ( ) NEW_LINE
print ( " Inorder ▁ traversal ▁ of ▁ the ▁ modified ▁ tree " ) NEW_LINE inorder ( root ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , x ) : NEW_LINE INDENT self . data = x NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def insert ( node , data ) : NEW_LINE INDENT global succ NEW_LINE DEDENT
root = node NEW_LINE if ( node == None ) : NEW_LINE INDENT return Node ( data ) NEW_LINE DEDENT
if ( data < node . data ) : NEW_LINE INDENT root . left = insert ( node . left , data ) NEW_LINE DEDENT elif ( data > node . data ) : NEW_LINE INDENT root . right = insert ( node . right , data ) NEW_LINE DEDENT
def check ( num ) : NEW_LINE INDENT sum = 0 NEW_LINE i = num NEW_LINE DEDENT
if ( num < 10 or num > 99 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT else : NEW_LINE INDENT sum_of_digits = ( i % 10 ) + ( i // 10 ) NEW_LINE prod_of_digits = ( i % 10 ) * ( i // 10 ) NEW_LINE sum = sum_of_digits + prod_of_digits NEW_LINE DEDENT if ( sum == num ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT else : NEW_LINE INDENT return 0 NEW_LINE DEDENT
def countSpecialDigit ( rt ) : NEW_LINE INDENT global c NEW_LINE if ( rt == None ) : NEW_LINE INDENT return NEW_LINE DEDENT else : NEW_LINE INDENT x = check ( rt . data ) NEW_LINE if ( x == 1 ) : NEW_LINE INDENT c += 1 NEW_LINE DEDENT countSpecialDigit ( rt . left ) NEW_LINE countSpecialDigit ( rt . right ) NEW_LINE DEDENT DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT root = None NEW_LINE c = 0 NEW_LINE root = insert ( root , 50 ) NEW_LINE root = insert ( root , 29 ) NEW_LINE root = insert ( root , 59 ) NEW_LINE root = insert ( root , 19 ) NEW_LINE root = insert ( root , 53 ) NEW_LINE root = insert ( root , 556 ) NEW_LINE root = insert ( root , 56 ) NEW_LINE root = insert ( root , 94 ) NEW_LINE root = insert ( root , 13 ) NEW_LINE DEDENT
countSpecialDigit ( root ) NEW_LINE print ( c ) NEW_LINE
def buildTree ( inorder , start , end ) : NEW_LINE INDENT if start > end : NEW_LINE INDENT return None NEW_LINE DEDENT DEDENT
i = Max ( inorder , start , end ) NEW_LINE
root = newNode ( inorder [ i ] ) NEW_LINE
if start == end : NEW_LINE INDENT return root NEW_LINE DEDENT
root . left = buildTree ( inorder , start , i - 1 ) NEW_LINE root . right = buildTree ( inorder , i + 1 , end ) NEW_LINE return root NEW_LINE
def Max ( arr , strt , end ) : NEW_LINE INDENT i , Max = 0 , arr [ strt ] NEW_LINE maxind = strt NEW_LINE for i in range ( strt + 1 , end + 1 ) : NEW_LINE INDENT if arr [ i ] > Max : NEW_LINE INDENT Max = arr [ i ] NEW_LINE maxind = i NEW_LINE DEDENT DEDENT return maxind NEW_LINE DEDENT
class newNode : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def printInorder ( node ) : NEW_LINE INDENT if node == None : NEW_LINE INDENT return NEW_LINE DEDENT DEDENT
printInorder ( node . left ) NEW_LINE
print ( node . data , end = " ▁ " ) NEW_LINE
printInorder ( node . right ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
inorder = [ 5 , 10 , 40 , 30 , 28 ] NEW_LINE Len = len ( inorder ) NEW_LINE root = buildTree ( inorder , 0 , Len - 1 ) NEW_LINE
print ( " Inorder ▁ traversal ▁ of ▁ the " , " constructed ▁ tree ▁ is ▁ " ) NEW_LINE printInorder ( root ) NEW_LINE
def Identity ( size ) : NEW_LINE INDENT for row in range ( 0 , size ) : NEW_LINE INDENT for col in range ( 0 , size ) : NEW_LINE DEDENT DEDENT
if ( row == col ) : NEW_LINE INDENT print ( "1 ▁ " , end = " ▁ " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( "0 ▁ " , end = " ▁ " ) NEW_LINE DEDENT print ( ) NEW_LINE
size = 5 NEW_LINE Identity ( size ) NEW_LINE
def search ( mat , n , x ) : NEW_LINE INDENT i = 0 NEW_LINE DEDENT
j = n - 1 NEW_LINE while ( i < n and j >= 0 ) : NEW_LINE INDENT if ( mat [ i ] [ j ] == x ) : NEW_LINE INDENT print ( " n ▁ Found ▁ at ▁ " , i , " , ▁ " , j ) NEW_LINE return 1 NEW_LINE DEDENT if ( mat [ i ] [ j ] > x ) : NEW_LINE INDENT j -= 1 NEW_LINE DEDENT DEDENT
else : NEW_LINE INDENT i += 1 NEW_LINE DEDENT print ( " Element ▁ not ▁ found " ) NEW_LINE
return 0 NEW_LINE
mat = [ [ 10 , 20 , 30 , 40 ] , [ 15 , 25 , 35 , 45 ] , [ 27 , 29 , 37 , 48 ] , [ 32 , 33 , 39 , 50 ] ] NEW_LINE search ( mat , 4 , 29 ) NEW_LINE
def fill0X ( m , n ) : NEW_LINE
i , k , l = 0 , 0 , 0 NEW_LINE
r = m NEW_LINE c = n NEW_LINE
a = [ [ None ] * n for i in range ( m ) ] NEW_LINE
x = ' X ' NEW_LINE
while k < m and l < n : NEW_LINE
for i in range ( l , n ) : NEW_LINE INDENT a [ k ] [ i ] = x NEW_LINE DEDENT k += 1 NEW_LINE
for i in range ( k , m ) : NEW_LINE INDENT a [ i ] [ n - 1 ] = x NEW_LINE DEDENT n -= 1 NEW_LINE
if k < m : NEW_LINE INDENT for i in range ( n - 1 , l - 1 , - 1 ) : NEW_LINE INDENT a [ m - 1 ] [ i ] = x NEW_LINE DEDENT m -= 1 NEW_LINE DEDENT
if l < n : NEW_LINE INDENT for i in range ( m - 1 , k - 1 , - 1 ) : NEW_LINE INDENT a [ i ] [ l ] = x NEW_LINE DEDENT l += 1 NEW_LINE DEDENT
x = ' X ' if x == '0' else '0' NEW_LINE
for i in range ( r ) : NEW_LINE INDENT for j in range ( c ) : NEW_LINE INDENT print ( a [ i ] [ j ] , end = " ▁ " ) NEW_LINE DEDENT print ( ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT print ( " Output ▁ for ▁ m ▁ = ▁ 5 , ▁ n ▁ = ▁ 6" ) NEW_LINE fill0X ( 5 , 6 ) NEW_LINE print ( " Output ▁ for ▁ m ▁ = ▁ 4 , ▁ n ▁ = ▁ 4" ) NEW_LINE fill0X ( 4 , 4 ) NEW_LINE print ( " Output ▁ for ▁ m ▁ = ▁ 3 , ▁ n ▁ = ▁ 4" ) NEW_LINE fill0X ( 3 , 4 ) NEW_LINE DEDENT
N = 3 ; NEW_LINE
def interchangeDiagonals ( array ) : NEW_LINE
for i in range ( N ) : NEW_LINE INDENT if ( i != N / 2 ) : NEW_LINE INDENT temp = array [ i ] [ i ] ; NEW_LINE array [ i ] [ i ] = array [ i ] [ N - i - 1 ] ; NEW_LINE array [ i ] [ N - i - 1 ] = temp ; NEW_LINE DEDENT DEDENT for i in range ( N ) : NEW_LINE INDENT for j in range ( N ) : NEW_LINE INDENT print ( array [ i ] [ j ] , end = " ▁ " ) ; NEW_LINE DEDENT print ( ) ; NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT array = [ 4 , 5 , 6 ] , [ 1 , 2 , 3 ] , [ 7 , 8 , 9 ] ; NEW_LINE interchangeDiagonals ( array ) ; NEW_LINE DEDENT
SIZE = 50 NEW_LINE
class node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . right = None NEW_LINE self . left = None NEW_LINE DEDENT DEDENT
class Queue : NEW_LINE INDENT def __init__ ( self ) : NEW_LINE INDENT self . front = None NEW_LINE self . rear = None NEW_LINE self . size = 0 NEW_LINE self . array = [ ] NEW_LINE DEDENT DEDENT
def newNode ( data ) : NEW_LINE INDENT temp = node ( data ) NEW_LINE return temp NEW_LINE DEDENT
def createQueue ( size ) : NEW_LINE INDENT global queue NEW_LINE queue = Queue ( ) ; NEW_LINE queue . front = queue . rear = - 1 ; NEW_LINE queue . size = size ; NEW_LINE queue . array = [ None for i in range ( size ) ] NEW_LINE return queue ; NEW_LINE DEDENT
def isEmpty ( queue ) : NEW_LINE INDENT return queue . front == - 1 NEW_LINE DEDENT def isFull ( queue ) : NEW_LINE INDENT return queue . rear == queue . size - 1 ; NEW_LINE DEDENT def hasOnlyOneItem ( queue ) : NEW_LINE INDENT return queue . front == queue . rear ; NEW_LINE DEDENT def Enqueue ( root ) : NEW_LINE INDENT if ( isFull ( queue ) ) : NEW_LINE INDENT return ; NEW_LINE DEDENT queue . rear += 1 NEW_LINE queue . array [ queue . rear ] = root ; NEW_LINE if ( isEmpty ( queue ) ) : NEW_LINE INDENT queue . front += 1 ; NEW_LINE DEDENT DEDENT def Dequeue ( ) : NEW_LINE INDENT if ( isEmpty ( queue ) ) : NEW_LINE INDENT return None ; NEW_LINE DEDENT temp = queue . array [ queue . front ] ; NEW_LINE if ( hasOnlyOneItem ( queue ) ) : NEW_LINE INDENT queue . front = queue . rear = - 1 ; NEW_LINE DEDENT else : NEW_LINE INDENT queue . front += 1 NEW_LINE DEDENT return temp ; NEW_LINE DEDENT def getFront ( queue ) : NEW_LINE INDENT return queue . array [ queue . front ] ; NEW_LINE DEDENT
def hasBothChild ( temp ) : NEW_LINE INDENT return ( temp and temp . left and temp . right ) ; NEW_LINE DEDENT
def insert ( root , data , queue ) : NEW_LINE
temp = newNode ( data ) ; NEW_LINE
if not root : NEW_LINE INDENT root = temp ; NEW_LINE DEDENT else : NEW_LINE
front = getFront ( queue ) ; NEW_LINE
if ( not front . left ) : NEW_LINE INDENT front . left = temp ; NEW_LINE DEDENT
elif ( not front . right ) : NEW_LINE INDENT front . right = temp ; NEW_LINE DEDENT
if ( hasBothChild ( front ) ) : NEW_LINE INDENT Dequeue ( ) ; NEW_LINE DEDENT
Enqueue ( temp ) ; NEW_LINE return root NEW_LINE
def levelOrder ( root ) : NEW_LINE INDENT queue = createQueue ( SIZE ) ; NEW_LINE Enqueue ( root ) ; NEW_LINE while ( not isEmpty ( queue ) ) : NEW_LINE INDENT temp = Dequeue ( ) ; NEW_LINE print ( temp . data , end = ' ▁ ' ) NEW_LINE if ( temp . left ) : NEW_LINE INDENT Enqueue ( temp . left ) ; NEW_LINE DEDENT if ( temp . right ) : NEW_LINE INDENT Enqueue ( temp . right ) ; NEW_LINE DEDENT DEDENT DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT root = None NEW_LINE queue = createQueue ( SIZE ) ; NEW_LINE for i in range ( 1 , 13 ) : NEW_LINE INDENT root = insert ( root , i , queue ) ; NEW_LINE DEDENT levelOrder ( root ) ; NEW_LINE DEDENT
class Node ( object ) : NEW_LINE INDENT def __init__ ( self , item ) : NEW_LINE INDENT self . data = item NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def BTToDLLUtil ( root ) : NEW_LINE
if root is None : NEW_LINE INDENT return root NEW_LINE DEDENT
if root . left : NEW_LINE
left = BTToDLLUtil ( root . left ) NEW_LINE
while left . right : NEW_LINE INDENT left = left . right NEW_LINE DEDENT
left . right = root NEW_LINE
root . left = left NEW_LINE
if root . right : NEW_LINE
right = BTToDLLUtil ( root . right ) NEW_LINE
while right . left : NEW_LINE INDENT right = right . left NEW_LINE DEDENT
right . left = root NEW_LINE
root . right = right NEW_LINE return root NEW_LINE
def BTToDLL ( root ) : NEW_LINE
if root is None : NEW_LINE INDENT return root NEW_LINE DEDENT
root = BTToDLLUtil ( root ) NEW_LINE
while root . left : NEW_LINE INDENT root = root . left NEW_LINE DEDENT return root NEW_LINE
def print_list ( head ) : NEW_LINE INDENT if head is None : NEW_LINE INDENT return NEW_LINE DEDENT while head : NEW_LINE INDENT print ( head . data , end = " ▁ " ) NEW_LINE head = head . right NEW_LINE DEDENT DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE
root = Node ( 10 ) NEW_LINE root . left = Node ( 12 ) NEW_LINE root . right = Node ( 15 ) NEW_LINE root . left . left = Node ( 25 ) NEW_LINE root . left . right = Node ( 30 ) NEW_LINE root . right . left = Node ( 36 ) NEW_LINE
head = BTToDLL ( root ) NEW_LINE
print_list ( head ) NEW_LINE
M = 4 NEW_LINE N = 5 NEW_LINE
def findCommon ( mat ) : NEW_LINE
column = [ N - 1 ] * M NEW_LINE
min_row = 0 NEW_LINE
while ( column [ min_row ] >= 0 ) : NEW_LINE
for i in range ( M ) : NEW_LINE INDENT if ( mat [ i ] [ column [ i ] ] < mat [ min_row ] [ column [ min_row ] ] ) : NEW_LINE INDENT min_row = i NEW_LINE DEDENT DEDENT
eq_count = 0 NEW_LINE
for i in range ( M ) : NEW_LINE
if ( mat [ i ] [ column [ i ] ] > mat [ min_row ] [ column [ min_row ] ] ) : NEW_LINE INDENT if ( column [ i ] == 0 ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT DEDENT
column [ i ] -= 1 NEW_LINE else : NEW_LINE eq_count += 1 NEW_LINE
if ( eq_count == M ) : NEW_LINE INDENT return mat [ min_row ] [ column [ min_row ] ] NEW_LINE DEDENT return - 1 NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT mat = [ [ 1 , 2 , 3 , 4 , 5 ] , [ 2 , 4 , 5 , 8 , 10 ] , [ 3 , 5 , 7 , 9 , 11 ] , [ 1 , 3 , 5 , 7 , 9 ] ] NEW_LINE result = findCommon ( mat ) NEW_LINE if ( result == - 1 ) : NEW_LINE INDENT print ( " No ▁ common ▁ element " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Common ▁ element ▁ is " , result ) NEW_LINE DEDENT DEDENT
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def inorder ( root ) : NEW_LINE INDENT if root is not None : NEW_LINE INDENT inorder ( root . left ) NEW_LINE print " TABSYMBOL % d " % ( root . data ) , NEW_LINE inorder ( root . right ) NEW_LINE DEDENT DEDENT
def fixPrevPtr ( root ) : NEW_LINE INDENT if root is not None : NEW_LINE INDENT fixPrevPtr ( root . left ) NEW_LINE root . left = fixPrevPtr . pre NEW_LINE fixPrevPtr . pre = root NEW_LINE fixPrevPtr ( root . right ) NEW_LINE DEDENT DEDENT
def fixNextPtr ( root ) : NEW_LINE INDENT prev = None NEW_LINE DEDENT
while ( root and root . right != None ) : NEW_LINE INDENT root = root . right NEW_LINE DEDENT
while ( root and root . left != None ) : NEW_LINE INDENT prev = root NEW_LINE root = root . left NEW_LINE root . right = prev NEW_LINE DEDENT
return root NEW_LINE
def BTToDLL ( root ) : NEW_LINE
fixPrevPtr ( root ) NEW_LINE
return fixNextPtr ( root ) NEW_LINE
def printList ( root ) : NEW_LINE INDENT while ( root != None ) : NEW_LINE INDENT print " TABSYMBOL % d " % ( root . data ) , NEW_LINE root = root . right NEW_LINE DEDENT DEDENT
root = Node ( 10 ) NEW_LINE root . left = Node ( 12 ) NEW_LINE root . right = Node ( 15 ) NEW_LINE root . left . left = Node ( 25 ) NEW_LINE root . left . right = Node ( 30 ) NEW_LINE root . right . left = Node ( 36 ) NEW_LINE print   " NEW_LINE INDENT Inorder Tree Traversal NEW_LINE DEDENT " NEW_LINE inorder ( root ) NEW_LINE fixPrevPtr . pre = None NEW_LINE head = BTToDLL ( root ) NEW_LINE print   " NEW_LINE INDENT DLL Traversal NEW_LINE DEDENT " NEW_LINE printList ( head ) NEW_LINE
def convertTree ( node ) : NEW_LINE INDENT left_data = 0 NEW_LINE right_data = 0 NEW_LINE diff = 0 NEW_LINE DEDENT
if ( node == None or ( node . left == None and node . right == None ) ) : NEW_LINE INDENT return NEW_LINE DEDENT else : NEW_LINE
convertTree ( node . left ) NEW_LINE convertTree ( node . right ) NEW_LINE
if ( node . left != None ) : NEW_LINE INDENT left_data = node . left . data NEW_LINE DEDENT
if ( node . right != None ) : NEW_LINE INDENT right_data = node . right . data NEW_LINE DEDENT
diff = left_data + right_data - node . data NEW_LINE
if ( diff > 0 ) : NEW_LINE INDENT node . data = node . data + diff NEW_LINE DEDENT
if ( diff < 0 ) : NEW_LINE
increment ( node , - diff ) NEW_LINE
def increment ( node , diff ) : NEW_LINE
if ( node . left != None ) : NEW_LINE INDENT node . left . data = node . left . data + diff NEW_LINE DEDENT
increment ( node . left , diff ) NEW_LINE
elif ( node . right != None ) : NEW_LINE INDENT node . right . data = node . right . data + diff NEW_LINE increment ( node . right , diff ) NEW_LINE DEDENT
increment ( node . right , diff ) NEW_LINE
def printInorder ( node ) : NEW_LINE INDENT if ( node == None ) : NEW_LINE INDENT return NEW_LINE DEDENT DEDENT
printInorder ( node . left ) NEW_LINE
print ( node . data , end = " ▁ " ) NEW_LINE
printInorder ( node . right ) NEW_LINE
class newNode : NEW_LINE INDENT def __init__ ( self , key ) : NEW_LINE INDENT self . data = key NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT root = newNode ( 50 ) NEW_LINE root . left = newNode ( 7 ) NEW_LINE root . right = newNode ( 2 ) NEW_LINE root . left . left = newNode ( 3 ) NEW_LINE root . left . right = newNode ( 5 ) NEW_LINE root . right . left = newNode ( 1 ) NEW_LINE root . right . right = newNode ( 30 ) NEW_LINE print ( " Inorder ▁ traversal ▁ before ▁ conversion " ) NEW_LINE printInorder ( root ) NEW_LINE convertTree ( root ) NEW_LINE print ( " Inorder traversal after conversion   " ) NEW_LINE printInorder ( root ) NEW_LINE DEDENT
class node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . left = None NEW_LINE self . right = None NEW_LINE self . data = data NEW_LINE DEDENT DEDENT
def toSumTree ( Node ) : NEW_LINE
if ( Node == None ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
old_val = Node . data NEW_LINE
Node . data = toSumTree ( Node . left ) + toSumTree ( Node . right ) NEW_LINE
return Node . data + old_val NEW_LINE
def printInorder ( Node ) : NEW_LINE INDENT if ( Node == None ) : NEW_LINE INDENT return NEW_LINE DEDENT printInorder ( Node . left ) NEW_LINE print ( Node . data , end = " ▁ " ) NEW_LINE printInorder ( Node . right ) NEW_LINE DEDENT
def newNode ( data ) : NEW_LINE INDENT temp = node ( 0 ) NEW_LINE temp . data = data NEW_LINE temp . left = None NEW_LINE temp . right = None NEW_LINE return temp NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT root = None NEW_LINE x = 0 NEW_LINE DEDENT
root = newNode ( 10 ) NEW_LINE root . left = newNode ( - 2 ) NEW_LINE root . right = newNode ( 6 ) NEW_LINE root . left . left = newNode ( 8 ) NEW_LINE root . left . right = newNode ( - 4 ) NEW_LINE root . right . left = newNode ( 7 ) NEW_LINE root . right . right = newNode ( 5 ) NEW_LINE toSumTree ( root ) NEW_LINE
print ( " Inorder ▁ Traversal ▁ of ▁ the ▁ resultant ▁ tree ▁ is : ▁ " ) NEW_LINE printInorder ( root ) NEW_LINE
def findPeakUtil ( arr , low , high , n ) : NEW_LINE
mid = low + ( high - low ) / 2 NEW_LINE mid = int ( mid ) NEW_LINE
if ( ( mid == 0 or arr [ mid - 1 ] <= arr [ mid ] ) and ( mid == n - 1 or arr [ mid + 1 ] <= arr [ mid ] ) ) : NEW_LINE INDENT return mid NEW_LINE DEDENT
elif ( mid > 0 and arr [ mid - 1 ] > arr [ mid ] ) : NEW_LINE INDENT return findPeakUtil ( arr , low , ( mid - 1 ) , n ) NEW_LINE DEDENT
else : NEW_LINE INDENT return findPeakUtil ( arr , ( mid + 1 ) , high , n ) NEW_LINE DEDENT
def findPeak ( arr , n ) : NEW_LINE INDENT return findPeakUtil ( arr , 0 , n - 1 , n ) NEW_LINE DEDENT
arr = [ 1 , 3 , 20 , 4 , 1 , 0 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Index ▁ of ▁ a ▁ peak ▁ point ▁ is " , findPeak ( arr , n ) ) NEW_LINE
def printRepeating ( arr , size ) : NEW_LINE INDENT print ( " Repeating ▁ elements ▁ are ▁ " , end = ' ' ) NEW_LINE for i in range ( 0 , size ) : NEW_LINE INDENT for j in range ( i + 1 , size ) : NEW_LINE INDENT if arr [ i ] == arr [ j ] : NEW_LINE INDENT print ( arr [ i ] , end = ' ▁ ' ) NEW_LINE DEDENT DEDENT DEDENT DEDENT
arr = [ 4 , 2 , 4 , 5 , 2 , 3 , 1 ] NEW_LINE arr_size = len ( arr ) NEW_LINE printRepeating ( arr , arr_size ) NEW_LINE
def printRepeating ( arr , size ) : NEW_LINE INDENT count = [ 0 ] * size NEW_LINE print ( " ▁ Repeating ▁ elements ▁ are ▁ " , end = " " ) NEW_LINE for i in range ( 0 , size ) : NEW_LINE INDENT if ( count [ arr [ i ] ] == 1 ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT else : NEW_LINE INDENT count [ arr [ i ] ] = count [ arr [ i ] ] + 1 NEW_LINE DEDENT DEDENT DEDENT
arr = [ 4 , 2 , 4 , 5 , 2 , 3 , 1 ] NEW_LINE arr_size = len ( arr ) NEW_LINE printRepeating ( arr , arr_size ) NEW_LINE
import math NEW_LINE
def printRepeating ( arr , size ) : NEW_LINE
S = 0 ; NEW_LINE
P = 1 ; NEW_LINE n = size - 2 NEW_LINE
for i in range ( 0 , size ) : NEW_LINE INDENT S = S + arr [ i ] NEW_LINE P = P * arr [ i ] NEW_LINE DEDENT
S = S - n * ( n + 1 ) // 2 NEW_LINE
P = P // fact ( n ) NEW_LINE
D = math . sqrt ( S * S - 4 * P ) NEW_LINE x = ( D + S ) // 2 NEW_LINE y = ( S - D ) // 2 NEW_LINE print ( " The ▁ two ▁ Repeating ▁ elements ▁ are ▁ " , ( int ) ( x ) , " ▁ & ▁ " , ( int ) ( y ) ) NEW_LINE
def fact ( n ) : NEW_LINE INDENT if ( n == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT else : NEW_LINE INDENT return ( n * fact ( n - 1 ) ) NEW_LINE DEDENT DEDENT
arr = [ 4 , 2 , 4 , 5 , 2 , 3 , 1 ] NEW_LINE arr_size = len ( arr ) NEW_LINE printRepeating ( arr , arr_size ) NEW_LINE
def printRepeating ( arr , size ) : NEW_LINE
xor = arr [ 0 ] NEW_LINE
n = size - 2 NEW_LINE x = 0 NEW_LINE y = 0 NEW_LINE
for i in range ( 1 , size ) : NEW_LINE INDENT xor ^= arr [ i ] NEW_LINE DEDENT for i in range ( 1 , n + 1 ) : NEW_LINE INDENT xor ^= i NEW_LINE DEDENT
set_bit_no = xor & ~ ( xor - 1 ) NEW_LINE
for i in range ( 0 , size ) : NEW_LINE INDENT if ( arr [ i ] & set_bit_no ) : NEW_LINE INDENT x = x ^ arr [ i ] NEW_LINE DEDENT DEDENT
else : NEW_LINE INDENT y = y ^ arr [ i ] NEW_LINE DEDENT
for i in range ( 1 , n + 1 ) : NEW_LINE INDENT if ( i & set_bit_no ) : NEW_LINE INDENT x = x ^ i NEW_LINE DEDENT DEDENT
else : NEW_LINE INDENT y = y ^ i NEW_LINE DEDENT
print ( " The ▁ two ▁ repeating " , " elements ▁ are " , y , x ) NEW_LINE
arr = [ 4 , 2 , 4 , 5 , 2 , 3 , 1 ] NEW_LINE arr_size = len ( arr ) NEW_LINE printRepeating ( arr , arr_size ) NEW_LINE
def printRepeating ( arr , size ) : NEW_LINE INDENT print ( " ▁ The ▁ repeating ▁ elements ▁ are " , end = " ▁ " ) NEW_LINE for i in range ( 0 , size ) : NEW_LINE INDENT if ( arr [ abs ( arr [ i ] ) ] > 0 ) : NEW_LINE INDENT arr [ abs ( arr [ i ] ) ] = ( - 1 ) * arr [ abs ( arr [ i ] ) ] NEW_LINE DEDENT else : NEW_LINE INDENT print ( abs ( arr [ i ] ) , end = " ▁ " ) NEW_LINE DEDENT DEDENT DEDENT
arr = [ 4 , 2 , 4 , 5 , 2 , 3 , 1 ] NEW_LINE arr_size = len ( arr ) NEW_LINE printRepeating ( arr , arr_size ) NEW_LINE
def subArraySum ( arr , n , sum_ ) : NEW_LINE
for i in range ( n ) : NEW_LINE INDENT curr_sum = arr [ i ] NEW_LINE DEDENT
j = i + 1 NEW_LINE while j <= n : NEW_LINE INDENT if curr_sum == sum_ : NEW_LINE INDENT print ( " Sum ▁ found ▁ between " ) NEW_LINE print ( " indexes ▁ % ▁ d ▁ and ▁ % ▁ d " % ( i , j - 1 ) ) NEW_LINE return 1 NEW_LINE DEDENT if curr_sum > sum_ or j == n : NEW_LINE INDENT break NEW_LINE DEDENT curr_sum = curr_sum + arr [ j ] NEW_LINE j += 1 NEW_LINE DEDENT print ( " No ▁ subarray ▁ found " ) NEW_LINE return 0 NEW_LINE
arr = [ 15 , 2 , 4 , 8 , 9 , 5 , 10 , 23 ] NEW_LINE n = len ( arr ) NEW_LINE sum_ = 23 NEW_LINE subArraySum ( arr , n , sum_ ) NEW_LINE
def subArraySum ( arr , n , sum_ ) : NEW_LINE
curr_sum = arr [ 0 ] NEW_LINE start = 0 NEW_LINE
i = 1 NEW_LINE while i <= n : NEW_LINE
while curr_sum > sum_ and start < i - 1 : NEW_LINE INDENT curr_sum = curr_sum - arr [ start ] NEW_LINE start += 1 NEW_LINE DEDENT
if curr_sum == sum_ : NEW_LINE INDENT print ( " Sum ▁ found ▁ between ▁ indexes " ) NEW_LINE print ( " % ▁ d ▁ and ▁ % ▁ d " % ( start , i - 1 ) ) NEW_LINE return 1 NEW_LINE DEDENT
if i < n : NEW_LINE INDENT curr_sum = curr_sum + arr [ i ] NEW_LINE DEDENT i += 1 NEW_LINE
print ( " No ▁ subarray ▁ found " ) NEW_LINE return 0 NEW_LINE
arr = [ 15 , 2 , 4 , 8 , 9 , 5 , 10 , 23 ] NEW_LINE n = len ( arr ) NEW_LINE sum_ = 23 NEW_LINE subArraySum ( arr , n , sum_ ) NEW_LINE
def find3Numbers ( A , arr_size , sum ) : NEW_LINE
for i in range ( 0 , arr_size - 2 ) : NEW_LINE
for j in range ( i + 1 , arr_size - 1 ) : NEW_LINE
for k in range ( j + 1 , arr_size ) : NEW_LINE INDENT if A [ i ] + A [ j ] + A [ k ] == sum : NEW_LINE INDENT print ( " Triplet ▁ is " , A [ i ] , " , ▁ " , A [ j ] , " , ▁ " , A [ k ] ) NEW_LINE return True NEW_LINE DEDENT DEDENT
return False NEW_LINE
A = [ 1 , 4 , 45 , 6 , 10 , 8 ] NEW_LINE sum = 22 NEW_LINE arr_size = len ( A ) NEW_LINE find3Numbers ( A , arr_size , sum ) NEW_LINE
def search ( arr , x ) : NEW_LINE INDENT for index , value in enumerate ( arr ) : NEW_LINE INDENT if value == x : NEW_LINE INDENT return index NEW_LINE DEDENT DEDENT return - 1 NEW_LINE DEDENT
arr = [ 1 , 10 , 30 , 15 ] NEW_LINE x = 30 NEW_LINE print ( x , " is ▁ present ▁ at ▁ index " , search ( arr , x ) ) NEW_LINE
def binarySearch ( arr , l , r , x ) : NEW_LINE INDENT if r >= l : NEW_LINE INDENT mid = l + ( r - l ) // 2 NEW_LINE DEDENT DEDENT
if arr [ mid ] == x : NEW_LINE INDENT return mid NEW_LINE DEDENT
elif arr [ mid ] > x : NEW_LINE INDENT return binarySearch ( arr , l , mid - 1 , x ) NEW_LINE DEDENT
else : NEW_LINE INDENT return binarySearch ( arr , mid + 1 , r , x ) NEW_LINE DEDENT else : NEW_LINE
return - 1 NEW_LINE
arr = [ 2 , 3 , 4 , 10 , 40 ] NEW_LINE x = 10 NEW_LINE result = binarySearch ( arr , 0 , len ( arr ) - 1 , x ) NEW_LINE if result != - 1 : NEW_LINE INDENT print ( " Element ▁ is ▁ present ▁ at ▁ index ▁ % ▁ d " % result ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Element ▁ is ▁ not ▁ present ▁ in ▁ array " ) NEW_LINE DEDENT
def binarySearch ( arr , l , r , x ) : NEW_LINE INDENT while l <= r : NEW_LINE INDENT mid = l + ( r - l ) // 2 ; NEW_LINE DEDENT DEDENT
if arr [ mid ] == x : NEW_LINE INDENT return mid NEW_LINE DEDENT
elif arr [ mid ] < x : NEW_LINE INDENT l = mid + 1 NEW_LINE DEDENT
else : NEW_LINE INDENT r = mid - 1 NEW_LINE DEDENT
return - 1 NEW_LINE
arr = [ 2 , 3 , 4 , 10 , 40 ] NEW_LINE x = 10 NEW_LINE result = binarySearch ( arr , 0 , len ( arr ) - 1 , x ) NEW_LINE if result != - 1 : NEW_LINE INDENT print ( " Element ▁ is ▁ present ▁ at ▁ index ▁ % ▁ d " % result ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Element ▁ is ▁ not ▁ present ▁ in ▁ array " ) NEW_LINE DEDENT
def interpolationSearch ( arr , lo , hi , x ) : NEW_LINE
if ( lo <= hi and x >= arr [ lo ] and x <= arr [ hi ] ) : NEW_LINE
pos = lo + ( ( hi - lo ) // ( arr [ hi ] - arr [ lo ] ) * ( x - arr [ lo ] ) ) NEW_LINE
if arr [ pos ] == x : NEW_LINE INDENT return pos NEW_LINE DEDENT
if arr [ pos ] < x : NEW_LINE INDENT return interpolationSearch ( arr , pos + 1 , hi , x ) NEW_LINE DEDENT
if arr [ pos ] > x : NEW_LINE INDENT return interpolationSearch ( arr , lo , pos - 1 , x ) NEW_LINE DEDENT return - 1 NEW_LINE
arr = [ 10 , 12 , 13 , 16 , 18 , 19 , 20 , 21 , 22 , 23 , 24 , 33 , 35 , 42 , 47 ] NEW_LINE n = len ( arr ) NEW_LINE
x = 18 NEW_LINE index = interpolationSearch ( arr , 0 , n - 1 , x ) NEW_LINE
if index != - 1 : NEW_LINE INDENT print ( " Element ▁ found ▁ at ▁ index " , index ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Element ▁ not ▁ found " ) NEW_LINE DEDENT
def partition ( arr , l , h ) : NEW_LINE INDENT i = ( l - 1 ) NEW_LINE x = arr [ h ] NEW_LINE for j in range ( l , h ) : NEW_LINE INDENT if arr [ j ] <= x : NEW_LINE INDENT i = i + 1 NEW_LINE arr [ i ] , arr [ j ] = arr [ j ] , arr [ i ] NEW_LINE DEDENT DEDENT arr [ i + 1 ] , arr [ h ] = arr [ h ] , arr [ i + 1 ] NEW_LINE return ( i + 1 ) NEW_LINE DEDENT
def quickSortIterative ( arr , l , h ) : NEW_LINE
size = h - l + 1 NEW_LINE stack = [ 0 ] * ( size ) NEW_LINE
top = - 1 NEW_LINE
top = top + 1 NEW_LINE stack [ top ] = l NEW_LINE top = top + 1 NEW_LINE stack [ top ] = h NEW_LINE
while top >= 0 : NEW_LINE
h = stack [ top ] NEW_LINE top = top - 1 NEW_LINE l = stack [ top ] NEW_LINE top = top - 1 NEW_LINE
p = partition ( arr , l , h ) NEW_LINE
if p - 1 > l : NEW_LINE INDENT top = top + 1 NEW_LINE stack [ top ] = l NEW_LINE top = top + 1 NEW_LINE stack [ top ] = p - 1 NEW_LINE DEDENT
if p + 1 < h : NEW_LINE INDENT top = top + 1 NEW_LINE stack [ top ] = p + 1 NEW_LINE top = top + 1 NEW_LINE stack [ top ] = h NEW_LINE DEDENT
arr = [ 4 , 3 , 5 , 2 , 1 , 3 , 2 , 3 ] NEW_LINE n = len ( arr ) NEW_LINE
quickSortIterative ( arr , 0 , n - 1 ) NEW_LINE print ( " Sorted ▁ array ▁ is : " ) NEW_LINE for i in range ( n ) : NEW_LINE INDENT print ( " % ▁ d " % arr [ i ] ) , NEW_LINE DEDENT
def printMaxActivities ( s , f ) : NEW_LINE INDENT n = len ( f ) NEW_LINE print " The ▁ following ▁ activities ▁ are ▁ selected " NEW_LINE DEDENT
i = 0 NEW_LINE print i , NEW_LINE
for j in xrange ( n ) : NEW_LINE
if s [ j ] >= f [ i ] : NEW_LINE INDENT print j , NEW_LINE i = j NEW_LINE DEDENT
s = [ 1 , 3 , 0 , 5 , 8 , 5 ] NEW_LINE f = [ 2 , 4 , 6 , 7 , 9 , 9 ] NEW_LINE printMaxActivities ( s , f ) NEW_LINE
def lcs ( X , Y , m , n ) : NEW_LINE INDENT if m == 0 or n == 0 : NEW_LINE return 0 ; NEW_LINE elif X [ m - 1 ] == Y [ n - 1 ] : NEW_LINE return 1 + lcs ( X , Y , m - 1 , n - 1 ) ; NEW_LINE else : NEW_LINE return max ( lcs ( X , Y , m , n - 1 ) , lcs ( X , Y , m - 1 , n ) ) ; NEW_LINE DEDENT
X = " AGGTAB " NEW_LINE Y = " GXTXAYB " NEW_LINE print " Length ▁ of ▁ LCS ▁ is ▁ " , lcs ( X , Y , len ( X ) , len ( Y ) ) NEW_LINE
def lcs ( X , Y ) : NEW_LINE
m = len ( X ) NEW_LINE n = len ( Y ) NEW_LINE L = [ [ None ] * ( n + 1 ) for i in xrange ( m + 1 ) ] NEW_LINE
for i in range ( m + 1 ) : NEW_LINE INDENT for j in range ( n + 1 ) : NEW_LINE INDENT if i == 0 or j == 0 : NEW_LINE INDENT L [ i ] [ j ] = 0 NEW_LINE DEDENT elif X [ i - 1 ] == Y [ j - 1 ] : NEW_LINE INDENT L [ i ] [ j ] = L [ i - 1 ] [ j - 1 ] + 1 NEW_LINE DEDENT else : NEW_LINE INDENT L [ i ] [ j ] = max ( L [ i - 1 ] [ j ] , L [ i ] [ j - 1 ] ) NEW_LINE DEDENT DEDENT DEDENT
return L [ m ] [ n ] NEW_LINE
X = " AGGTAB " NEW_LINE Y = " GXTXAYB " NEW_LINE print " Length ▁ of ▁ LCS ▁ is ▁ " , lcs ( X , Y ) NEW_LINE
R = 3 NEW_LINE C = 3 NEW_LINE import sys NEW_LINE
def min ( x , y , z ) : NEW_LINE INDENT if ( x < y ) : NEW_LINE INDENT return x if ( x < z ) else z NEW_LINE DEDENT else : NEW_LINE INDENT return y if ( y < z ) else z NEW_LINE DEDENT DEDENT
def minCost ( cost , m , n ) : NEW_LINE INDENT if ( n < 0 or m < 0 ) : NEW_LINE INDENT return sys . maxsize NEW_LINE DEDENT elif ( m == 0 and n == 0 ) : NEW_LINE INDENT return cost [ m ] [ n ] NEW_LINE DEDENT else : NEW_LINE INDENT return cost [ m ] [ n ] + min ( minCost ( cost , m - 1 , n - 1 ) , minCost ( cost , m - 1 , n ) , minCost ( cost , m , n - 1 ) ) NEW_LINE DEDENT DEDENT
cost = [ [ 1 , 2 , 3 ] , [ 4 , 8 , 2 ] , [ 1 , 5 , 3 ] ] NEW_LINE print ( minCost ( cost , 2 , 2 ) ) NEW_LINE
R = 3 NEW_LINE C = 3 NEW_LINE def minCost ( cost , m , n ) : NEW_LINE
tc = [ [ 0 for x in range ( C ) ] for x in range ( R ) ] NEW_LINE tc [ 0 ] [ 0 ] = cost [ 0 ] [ 0 ] NEW_LINE
for i in range ( 1 , m + 1 ) : NEW_LINE INDENT tc [ i ] [ 0 ] = tc [ i - 1 ] [ 0 ] + cost [ i ] [ 0 ] NEW_LINE DEDENT
for j in range ( 1 , n + 1 ) : NEW_LINE INDENT tc [ 0 ] [ j ] = tc [ 0 ] [ j - 1 ] + cost [ 0 ] [ j ] NEW_LINE DEDENT
for i in range ( 1 , m + 1 ) : NEW_LINE INDENT for j in range ( 1 , n + 1 ) : NEW_LINE INDENT tc [ i ] [ j ] = min ( tc [ i - 1 ] [ j - 1 ] , tc [ i - 1 ] [ j ] , tc [ i ] [ j - 1 ] ) + cost [ i ] [ j ] NEW_LINE DEDENT DEDENT return tc [ m ] [ n ] NEW_LINE
cost = [ [ 1 , 2 , 3 ] , [ 4 , 8 , 2 ] , [ 1 , 5 , 3 ] ] NEW_LINE print ( minCost ( cost , 2 , 2 ) ) NEW_LINE
def knapSack ( W , wt , val , n ) : NEW_LINE
if n == 0 or W == 0 : NEW_LINE INDENT return 0 NEW_LINE DEDENT
if ( wt [ n - 1 ] > W ) : NEW_LINE INDENT return knapSack ( W , wt , val , n - 1 ) NEW_LINE DEDENT
else : NEW_LINE INDENT return max ( val [ n - 1 ] + knapSack ( W - wt [ n - 1 ] , wt , val , n - 1 ) , knapSack ( W , wt , val , n - 1 ) ) NEW_LINE DEDENT
val = [ 60 , 100 , 120 ] NEW_LINE wt = [ 10 , 20 , 30 ] NEW_LINE W = 50 NEW_LINE n = len ( val ) NEW_LINE print knapSack ( W , wt , val , n ) NEW_LINE
def knapSack ( W , wt , val , n ) : NEW_LINE INDENT K = [ [ 0 for x in range ( W + 1 ) ] for x in range ( n + 1 ) ] NEW_LINE DEDENT
for i in range ( n + 1 ) : NEW_LINE INDENT for w in range ( W + 1 ) : NEW_LINE INDENT if i == 0 or w == 0 : NEW_LINE INDENT K [ i ] [ w ] = 0 NEW_LINE DEDENT elif wt [ i - 1 ] <= w : NEW_LINE INDENT K [ i ] [ w ] = max ( val [ i - 1 ] + K [ i - 1 ] [ w - wt [ i - 1 ] ] , K [ i - 1 ] [ w ] ) NEW_LINE DEDENT else : NEW_LINE INDENT K [ i ] [ w ] = K [ i - 1 ] [ w ] NEW_LINE DEDENT DEDENT DEDENT return K [ n ] [ W ] NEW_LINE
val = [ 60 , 100 , 120 ] NEW_LINE wt = [ 10 , 20 , 30 ] NEW_LINE W = 50 NEW_LINE n = len ( val ) NEW_LINE print ( knapSack ( W , wt , val , n ) ) NEW_LINE
import sys NEW_LINE
def eggDrop ( n , k ) : NEW_LINE
if ( k == 1 or k == 0 ) : NEW_LINE INDENT return k NEW_LINE DEDENT
if ( n == 1 ) : NEW_LINE INDENT return k NEW_LINE DEDENT min = sys . maxsize NEW_LINE
for x in range ( 1 , k + 1 ) : NEW_LINE INDENT res = max ( eggDrop ( n - 1 , x - 1 ) , eggDrop ( n , k - x ) ) NEW_LINE if ( res < min ) : NEW_LINE INDENT min = res NEW_LINE DEDENT DEDENT return min + 1 NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 2 NEW_LINE k = 10 NEW_LINE print ( " Minimum ▁ number ▁ of ▁ trials ▁ in ▁ worst ▁ case ▁ with " , n , " eggs ▁ and " , k , " floors ▁ is " , eggDrop ( n , k ) ) NEW_LINE DEDENT
def max ( x , y ) : NEW_LINE INDENT if ( x > y ) : NEW_LINE INDENT return x NEW_LINE DEDENT return y NEW_LINE DEDENT
def lps ( seq , i , j ) : NEW_LINE
if ( i == j ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT
if ( seq [ i ] == seq [ j ] and i + 1 == j ) : NEW_LINE INDENT return 2 NEW_LINE DEDENT
if ( seq [ i ] == seq [ j ] ) : NEW_LINE INDENT return lps ( seq , i + 1 , j - 1 ) + 2 NEW_LINE DEDENT
return max ( lps ( seq , i , j - 1 ) , lps ( seq , i + 1 , j ) ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT seq = " GEEKSFORGEEKS " NEW_LINE n = len ( seq ) NEW_LINE print ( " The ▁ length ▁ of ▁ the ▁ LPS ▁ is " , lps ( seq , 0 , n - 1 ) ) NEW_LINE DEDENT
INF = 2147483647 NEW_LINE
def printSolution ( p , n ) : NEW_LINE INDENT k = 0 NEW_LINE if p [ n ] == 1 : NEW_LINE INDENT k = 1 NEW_LINE DEDENT else : NEW_LINE INDENT k = printSolution ( p , p [ n ] - 1 ) + 1 NEW_LINE DEDENT print ( ' Line ▁ number ▁ ' , k , ' : ▁ From ▁ word ▁ no . ▁ ' , p [ n ] , ' to ▁ ' , n ) NEW_LINE return k NEW_LINE DEDENT
def solveWordWrap ( l , n , M ) : NEW_LINE
extras = [ [ 0 for i in range ( n + 1 ) ] for i in range ( n + 1 ) ] NEW_LINE
lc = [ [ 0 for i in range ( n + 1 ) ] for i in range ( n + 1 ) ] NEW_LINE
c = [ 0 for i in range ( n + 1 ) ] NEW_LINE
p = [ 0 for i in range ( n + 1 ) ] NEW_LINE
for i in range ( n + 1 ) : NEW_LINE INDENT extras [ i ] [ i ] = M - l [ i - 1 ] NEW_LINE for j in range ( i + 1 , n + 1 ) : NEW_LINE INDENT extras [ i ] [ j ] = ( extras [ i ] [ j - 1 ] - l [ j - 1 ] - 1 ) NEW_LINE DEDENT DEDENT
for i in range ( n + 1 ) : NEW_LINE INDENT for j in range ( i , n + 1 ) : NEW_LINE INDENT if extras [ i ] [ j ] < 0 : NEW_LINE INDENT lc [ i ] [ j ] = INF ; NEW_LINE DEDENT elif j == n and extras [ i ] [ j ] >= 0 : NEW_LINE INDENT lc [ i ] [ j ] = 0 NEW_LINE DEDENT else : NEW_LINE INDENT lc [ i ] [ j ] = ( extras [ i ] [ j ] * extras [ i ] [ j ] ) NEW_LINE DEDENT DEDENT DEDENT
c [ 0 ] = 0 NEW_LINE for j in range ( 1 , n + 1 ) : NEW_LINE INDENT c [ j ] = INF NEW_LINE for i in range ( 1 , j + 1 ) : NEW_LINE INDENT if ( c [ i - 1 ] != INF and lc [ i ] [ j ] != INF and ( ( c [ i - 1 ] + lc [ i ] [ j ] ) < c [ j ] ) ) : NEW_LINE INDENT c [ j ] = c [ i - 1 ] + lc [ i ] [ j ] NEW_LINE p [ j ] = i NEW_LINE DEDENT DEDENT DEDENT printSolution ( p , n ) NEW_LINE
l = [ 3 , 2 , 2 , 5 ] NEW_LINE n = len ( l ) NEW_LINE M = 6 NEW_LINE solveWordWrap ( l , n , M ) NEW_LINE
def Sum ( freq , i , j ) : NEW_LINE INDENT s = 0 NEW_LINE for k in range ( i , j + 1 ) : NEW_LINE INDENT s += freq [ k ] NEW_LINE DEDENT return s NEW_LINE DEDENT
def optCost ( freq , i , j ) : NEW_LINE
if j < i : NEW_LINE INDENT return 0 NEW_LINE DEDENT
if j == i : NEW_LINE INDENT return freq [ i ] NEW_LINE DEDENT
fsum = Sum ( freq , i , j ) NEW_LINE
Min = 999999999999 NEW_LINE
for r in range ( i , j + 1 ) : NEW_LINE INDENT cost = ( optCost ( freq , i , r - 1 ) + optCost ( freq , r + 1 , j ) ) NEW_LINE if cost < Min : NEW_LINE INDENT Min = cost NEW_LINE DEDENT DEDENT
return Min + fsum NEW_LINE
def optimalSearchTree ( keys , freq , n ) : NEW_LINE
return optCost ( freq , 0 , n - 1 ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT keys = [ 10 , 12 , 20 ] NEW_LINE freq = [ 34 , 8 , 50 ] NEW_LINE n = len ( keys ) NEW_LINE print ( " Cost ▁ of ▁ Optimal ▁ BST ▁ is " , optimalSearchTree ( keys , freq , n ) ) NEW_LINE DEDENT
INT_MAX = 2147483647 NEW_LINE
def sum ( freq , i , j ) : NEW_LINE INDENT s = 0 NEW_LINE for k in range ( i , j + 1 ) : NEW_LINE INDENT s += freq [ k ] NEW_LINE DEDENT return s NEW_LINE DEDENT
def optimalSearchTree ( keys , freq , n ) : NEW_LINE
cost = [ [ 0 for x in range ( n ) ] for y in range ( n ) ] NEW_LINE
for i in range ( n ) : NEW_LINE INDENT cost [ i ] [ i ] = freq [ i ] NEW_LINE DEDENT
for L in range ( 2 , n + 1 ) : NEW_LINE
for i in range ( n - L + 2 ) : NEW_LINE
j = i + L - 1 NEW_LINE if i >= n or j >= n : NEW_LINE INDENT break NEW_LINE DEDENT cost [ i ] [ j ] = INT_MAX NEW_LINE
for r in range ( i , j + 1 ) : NEW_LINE
c = 0 NEW_LINE if ( r > i ) : NEW_LINE INDENT c += cost [ i ] [ r - 1 ] NEW_LINE DEDENT if ( r < j ) : NEW_LINE INDENT c += cost [ r + 1 ] [ j ] NEW_LINE DEDENT c += sum ( freq , i , j ) NEW_LINE if ( c < cost [ i ] [ j ] ) : NEW_LINE INDENT cost [ i ] [ j ] = c NEW_LINE DEDENT return cost [ 0 ] [ n - 1 ] NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT keys = [ 10 , 12 , 20 ] NEW_LINE freq = [ 34 , 8 , 50 ] NEW_LINE n = len ( keys ) NEW_LINE print ( " Cost ▁ of ▁ Optimal ▁ BST ▁ is " , optimalSearchTree ( keys , freq , n ) ) NEW_LINE DEDENT
def max ( x , y ) : NEW_LINE INDENT if ( x > y ) : NEW_LINE INDENT return x NEW_LINE DEDENT else : NEW_LINE INDENT return y NEW_LINE DEDENT DEDENT
class node : NEW_LINE INDENT def __init__ ( self ) : NEW_LINE INDENT self . data = 0 NEW_LINE self . left = self . right = None NEW_LINE DEDENT DEDENT
def LISS ( root ) : NEW_LINE INDENT if ( root == None ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT DEDENT
size_excl = LISS ( root . left ) + LISS ( root . right ) NEW_LINE
size_incl = 1 NEW_LINE if ( root . left != None ) : NEW_LINE INDENT size_incl += LISS ( root . left . left ) + LISS ( root . left . right ) NEW_LINE DEDENT if ( root . right != None ) : NEW_LINE INDENT size_incl += LISS ( root . right . left ) + LISS ( root . right . right ) NEW_LINE DEDENT
return max ( size_incl , size_excl ) NEW_LINE
def newNode ( data ) : NEW_LINE INDENT temp = node ( ) NEW_LINE temp . data = data NEW_LINE temp . left = temp . right = None NEW_LINE return temp NEW_LINE DEDENT
root = newNode ( 20 ) NEW_LINE root . left = newNode ( 8 ) NEW_LINE root . left . left = newNode ( 4 ) NEW_LINE root . left . right = newNode ( 12 ) NEW_LINE root . left . right . left = newNode ( 10 ) NEW_LINE root . left . right . right = newNode ( 14 ) NEW_LINE root . right = newNode ( 22 ) NEW_LINE root . right . right = newNode ( 25 ) NEW_LINE print ( " Size ▁ of ▁ the ▁ Largest " , " ▁ Independent ▁ Set ▁ is ▁ " , LISS ( root ) ) NEW_LINE
def getCount ( keypad , n ) : NEW_LINE INDENT if ( not keypad or n <= 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT if ( n == 1 ) : NEW_LINE INDENT return 10 NEW_LINE DEDENT DEDENT
odd = [ 0 ] * 10 NEW_LINE even = [ 0 ] * 10 NEW_LINE i = 0 NEW_LINE j = 0 NEW_LINE useOdd = 0 NEW_LINE totalCount = 0 NEW_LINE for i in range ( 10 ) : NEW_LINE
odd [ i ] = 1 NEW_LINE
for j in range ( 2 , n + 1 ) : NEW_LINE INDENT useOdd = 1 - useOdd NEW_LINE DEDENT
if ( useOdd == 1 ) : NEW_LINE INDENT even [ 0 ] = odd [ 0 ] + odd [ 8 ] NEW_LINE even [ 1 ] = odd [ 1 ] + odd [ 2 ] + odd [ 4 ] NEW_LINE even [ 2 ] = odd [ 2 ] + odd [ 1 ] + odd [ 3 ] + odd [ 5 ] NEW_LINE even [ 3 ] = odd [ 3 ] + odd [ 2 ] + odd [ 6 ] NEW_LINE even [ 4 ] = odd [ 4 ] + odd [ 1 ] + odd [ 5 ] + odd [ 7 ] NEW_LINE even [ 5 ] = odd [ 5 ] + odd [ 2 ] + odd [ 4 ] + odd [ 8 ] + odd [ 6 ] NEW_LINE even [ 6 ] = odd [ 6 ] + odd [ 3 ] + odd [ 5 ] + odd [ 9 ] NEW_LINE even [ 7 ] = odd [ 7 ] + odd [ 4 ] + odd [ 8 ] NEW_LINE even [ 8 ] = odd [ 8 ] + odd [ 0 ] + odd [ 5 ] + odd [ 7 ] + odd [ 9 ] NEW_LINE even [ 9 ] = odd [ 9 ] + odd [ 6 ] + odd [ 8 ] NEW_LINE DEDENT else : NEW_LINE INDENT odd [ 0 ] = even [ 0 ] + even [ 8 ] NEW_LINE odd [ 1 ] = even [ 1 ] + even [ 2 ] + even [ 4 ] NEW_LINE odd [ 2 ] = even [ 2 ] + even [ 1 ] + even [ 3 ] + even [ 5 ] NEW_LINE odd [ 3 ] = even [ 3 ] + even [ 2 ] + even [ 6 ] NEW_LINE odd [ 4 ] = even [ 4 ] + even [ 1 ] + even [ 5 ] + even [ 7 ] NEW_LINE odd [ 5 ] = even [ 5 ] + even [ 2 ] + even [ 4 ] + even [ 8 ] + even [ 6 ] NEW_LINE odd [ 6 ] = even [ 6 ] + even [ 3 ] + even [ 5 ] + even [ 9 ] NEW_LINE odd [ 7 ] = even [ 7 ] + even [ 4 ] + even [ 8 ] NEW_LINE odd [ 8 ] = even [ 8 ] + even [ 0 ] + even [ 5 ] + even [ 7 ] + even [ 9 ] NEW_LINE odd [ 9 ] = even [ 9 ] + even [ 6 ] + even [ 8 ] NEW_LINE DEDENT
totalCount = 0 NEW_LINE if ( useOdd == 1 ) : NEW_LINE INDENT for i in range ( 10 ) : NEW_LINE INDENT totalCount += even [ i ] NEW_LINE DEDENT DEDENT else : NEW_LINE INDENT for i in range ( 10 ) : NEW_LINE INDENT totalCount += odd [ i ] NEW_LINE DEDENT DEDENT return totalCount NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT keypad = [ [ '1' , '2' , '3' ] , [ '4' , '5' , '6' ] , [ '7' , '8' , '9' ] , [ ' * ' , '0' , ' # ' ] ] NEW_LINE print ( " Count ▁ for ▁ numbers ▁ of ▁ length ▁ " , 1 , " : ▁ " , getCount ( keypad , 1 ) ) NEW_LINE print ( " Count ▁ for ▁ numbers ▁ of ▁ length ▁ " , 2 , " : ▁ " , getCount ( keypad , 2 ) ) NEW_LINE print ( " Count ▁ for ▁ numbers ▁ of ▁ length ▁ " , 3 , " : ▁ " , getCount ( keypad , 3 ) ) NEW_LINE print ( " Count ▁ for ▁ numbers ▁ of ▁ length ▁ " , 4 , " : ▁ " , getCount ( keypad , 4 ) ) NEW_LINE print ( " Count ▁ for ▁ numbers ▁ of ▁ length ▁ " , 5 , " : ▁ " , getCount ( keypad , 5 ) ) NEW_LINE DEDENT
def count ( n ) : NEW_LINE
table = [ 0 for i in range ( n + 1 ) ] NEW_LINE
table [ 0 ] = 1 NEW_LINE
for i in range ( 3 , n + 1 ) : NEW_LINE INDENT table [ i ] += table [ i - 3 ] NEW_LINE DEDENT for i in range ( 5 , n + 1 ) : NEW_LINE INDENT table [ i ] += table [ i - 5 ] NEW_LINE DEDENT for i in range ( 10 , n + 1 ) : NEW_LINE INDENT table [ i ] += table [ i - 10 ] NEW_LINE DEDENT return table [ n ] NEW_LINE
n = 20 NEW_LINE print ( ' Count ▁ for ' , n , ' is ' , count ( n ) ) NEW_LINE n = 13 NEW_LINE print ( ' Count ▁ for ' , n , ' is ' , count ( n ) ) NEW_LINE
def search ( pat , txt ) : NEW_LINE INDENT M = len ( pat ) NEW_LINE N = len ( txt ) NEW_LINE DEDENT
for i in range ( N - M + 1 ) : NEW_LINE INDENT j = 0 NEW_LINE DEDENT
while ( j < M ) : NEW_LINE INDENT if ( txt [ i + j ] != pat [ j ] ) : NEW_LINE INDENT break NEW_LINE DEDENT j += 1 NEW_LINE DEDENT
if ( j == M ) : NEW_LINE INDENT print ( " Pattern ▁ found ▁ at ▁ index ▁ " , i ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT txt = " AABAACAADAABAAABAA " NEW_LINE pat = " AABA " NEW_LINE search ( pat , txt ) NEW_LINE DEDENT
d = 256 NEW_LINE
def search ( pat , txt , q ) : NEW_LINE INDENT M = len ( pat ) NEW_LINE N = len ( txt ) NEW_LINE i = 0 NEW_LINE j = 0 NEW_LINE DEDENT
p = 0 NEW_LINE
t = 0 NEW_LINE h = 1 NEW_LINE
for i in xrange ( M - 1 ) : NEW_LINE INDENT h = ( h * d ) % q NEW_LINE DEDENT
for i in xrange ( M ) : NEW_LINE INDENT p = ( d * p + ord ( pat [ i ] ) ) % q NEW_LINE t = ( d * t + ord ( txt [ i ] ) ) % q NEW_LINE DEDENT
for i in xrange ( N - M + 1 ) : NEW_LINE
if p == t : NEW_LINE
for j in xrange ( M ) : NEW_LINE INDENT if txt [ i + j ] != pat [ j ] : NEW_LINE INDENT break NEW_LINE DEDENT else : j += 1 NEW_LINE DEDENT
if j == M : NEW_LINE INDENT print " Pattern ▁ found ▁ at ▁ index ▁ " + str ( i ) NEW_LINE DEDENT
if i < N - M : NEW_LINE INDENT t = ( d * ( t - ord ( txt [ i ] ) * h ) + ord ( txt [ i + M ] ) ) % q NEW_LINE DEDENT
if t < 0 : NEW_LINE INDENT t = t + q NEW_LINE DEDENT
txt = " GEEKS ▁ FOR ▁ GEEKS " NEW_LINE pat = " GEEK " NEW_LINE
q = 101 NEW_LINE
search ( pat , txt , q ) NEW_LINE
def search ( pat , txt ) : NEW_LINE INDENT M = len ( pat ) NEW_LINE N = len ( txt ) NEW_LINE i = 0 NEW_LINE while i <= N - M : NEW_LINE DEDENT
for j in xrange ( M ) : NEW_LINE INDENT if txt [ i + j ] != pat [ j ] : NEW_LINE INDENT break NEW_LINE DEDENT j += 1 NEW_LINE DEDENT
if j == M : NEW_LINE INDENT print " Pattern ▁ found ▁ at ▁ index ▁ " + str ( i ) NEW_LINE i = i + M NEW_LINE DEDENT elif j == 0 : NEW_LINE INDENT i = i + 1 NEW_LINE DEDENT else : NEW_LINE
i = i + j NEW_LINE
txt = " ABCEABCDABCEABCD " NEW_LINE pat = " ABCD " NEW_LINE search ( pat , txt ) NEW_LINE
NO_OF_CHARS = 256 NEW_LINE def getNextState ( pat , M , state , x ) : NEW_LINE
if state < M and x == ord ( pat [ state ] ) : NEW_LINE INDENT return state + 1 NEW_LINE DEDENT i = 0 NEW_LINE
for ns in range ( state , 0 , - 1 ) : NEW_LINE INDENT if ord ( pat [ ns - 1 ] ) == x : NEW_LINE INDENT while ( i < ns - 1 ) : NEW_LINE INDENT if pat [ i ] != pat [ state - ns + 1 + i ] : NEW_LINE INDENT break NEW_LINE DEDENT i += 1 NEW_LINE DEDENT if i == ns - 1 : NEW_LINE INDENT return ns NEW_LINE DEDENT DEDENT DEDENT return 0 NEW_LINE
def computeTF ( pat , M ) : NEW_LINE INDENT global NO_OF_CHARS NEW_LINE TF = [ [ 0 for i in range ( NO_OF_CHARS ) ] \ for _ in range ( M + 1 ) ] NEW_LINE for state in range ( M + 1 ) : NEW_LINE INDENT for x in range ( NO_OF_CHARS ) : NEW_LINE INDENT z = getNextState ( pat , M , state , x ) NEW_LINE TF [ state ] [ x ] = z NEW_LINE DEDENT DEDENT return TF NEW_LINE DEDENT
def search ( pat , txt ) : NEW_LINE INDENT global NO_OF_CHARS NEW_LINE M = len ( pat ) NEW_LINE N = len ( txt ) NEW_LINE TF = computeTF ( pat , M ) NEW_LINE DEDENT
state = 0 NEW_LINE for i in range ( N ) : NEW_LINE INDENT state = TF [ state ] [ ord ( txt [ i ] ) ] NEW_LINE if state == M : NEW_LINE INDENT print ( " Pattern ▁ found ▁ at ▁ index : ▁ { } " . format ( i - M + 1 ) ) NEW_LINE DEDENT DEDENT
def main ( ) : NEW_LINE INDENT txt = " AABAACAADAABAAABAA " NEW_LINE pat = " AABA " NEW_LINE search ( pat , txt ) NEW_LINE DEDENT if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT main ( ) NEW_LINE DEDENT
NO_OF_CHARS = 256 NEW_LINE
def badCharHeuristic ( string , size ) : NEW_LINE
badChar = [ - 1 ] * NO_OF_CHARS NEW_LINE
for i in range ( size ) : NEW_LINE INDENT badChar [ ord ( string [ i ] ) ] = i ; NEW_LINE DEDENT return badChar NEW_LINE
def search ( txt , pat ) : NEW_LINE INDENT m = len ( pat ) NEW_LINE n = len ( txt ) NEW_LINE DEDENT
badChar = badCharHeuristic ( pat , m ) NEW_LINE
s = 0 NEW_LINE
while ( s <= n - m ) : NEW_LINE INDENT j = m - 1 NEW_LINE DEDENT
while j >= 0 and pat [ j ] == txt [ s + j ] : NEW_LINE INDENT j -= 1 NEW_LINE DEDENT
if j < 0 : NEW_LINE INDENT print ( " Pattern ▁ occur ▁ at ▁ shift ▁ = ▁ { } " . format ( s ) ) NEW_LINE DEDENT
s += ( m - badChar [ ord ( txt [ s + m ] ) ] if s + m < n else 1 ) NEW_LINE else : NEW_LINE
s += max ( 1 , j - badChar [ ord ( txt [ s + j ] ) ] ) NEW_LINE
def main ( ) : NEW_LINE INDENT txt = " ABAAABCD " NEW_LINE pat = " ABC " NEW_LINE search ( txt , pat ) NEW_LINE DEDENT if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT main ( ) NEW_LINE DEDENT
N = 9 NEW_LINE
def printing ( arr ) : NEW_LINE INDENT for i in range ( N ) : NEW_LINE INDENT for j in range ( N ) : NEW_LINE INDENT print ( arr [ i ] [ j ] , end = " ▁ " ) NEW_LINE DEDENT print ( ) NEW_LINE DEDENT DEDENT
def isSafe ( grid , row , col , num ) : NEW_LINE
for x in range ( 9 ) : NEW_LINE INDENT if grid [ row ] [ x ] == num : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT
for x in range ( 9 ) : NEW_LINE INDENT if grid [ x ] [ col ] == num : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT
startRow = row - row % 3 NEW_LINE startCol = col - col % 3 NEW_LINE for i in range ( 3 ) : NEW_LINE INDENT for j in range ( 3 ) : NEW_LINE INDENT if grid [ i + startRow ] [ j + startCol ] == num : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT DEDENT return True NEW_LINE
def solveSuduko ( grid , row , col ) : NEW_LINE
if ( row == N - 1 and col == N ) : NEW_LINE INDENT return True NEW_LINE DEDENT
if col == N : NEW_LINE INDENT row += 1 NEW_LINE col = 0 NEW_LINE DEDENT
if grid [ row ] [ col ] > 0 : NEW_LINE INDENT return solveSuduko ( grid , row , col + 1 ) NEW_LINE DEDENT for num in range ( 1 , N + 1 , 1 ) : NEW_LINE
if isSafe ( grid , row , col , num ) : NEW_LINE
grid [ row ] [ col ] = num NEW_LINE
if solveSuduko ( grid , row , col + 1 ) : NEW_LINE INDENT return True NEW_LINE DEDENT
grid [ row ] [ col ] = 0 NEW_LINE return False NEW_LINE
grid = [ [ 3 , 0 , 6 , 5 , 0 , 8 , 4 , 0 , 0 ] , [ 5 , 2 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ] , [ 0 , 8 , 7 , 0 , 0 , 0 , 0 , 3 , 1 ] , [ 0 , 0 , 3 , 0 , 1 , 0 , 0 , 8 , 0 ] , [ 9 , 0 , 0 , 8 , 6 , 3 , 0 , 0 , 5 ] , [ 0 , 5 , 0 , 0 , 9 , 0 , 6 , 0 , 0 ] , [ 1 , 3 , 0 , 0 , 0 , 0 , 2 , 5 , 0 ] , [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 7 , 4 ] , [ 0 , 0 , 5 , 2 , 0 , 6 , 3 , 0 , 0 ] ] NEW_LINE if ( solveSuduko ( grid , 0 , 0 ) ) : NEW_LINE INDENT printing ( grid ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " no ▁ solution ▁ exists ▁ " ) NEW_LINE DEDENT
def getMedian ( ar1 , ar2 , n ) : NEW_LINE INDENT i = 0 NEW_LINE j = 0 NEW_LINE m1 = - 1 NEW_LINE m2 = - 1 NEW_LINE DEDENT
count = 0 NEW_LINE while count < n + 1 : NEW_LINE INDENT count += 1 NEW_LINE DEDENT
if i == n : NEW_LINE INDENT m1 = m2 NEW_LINE m2 = ar2 [ 0 ] NEW_LINE break NEW_LINE DEDENT
elif j == n : NEW_LINE INDENT m1 = m2 NEW_LINE m2 = ar1 [ 0 ] NEW_LINE break NEW_LINE DEDENT
if ar1 [ i ] <= ar2 [ j ] : NEW_LINE
m1 = m2 NEW_LINE m2 = ar1 [ i ] NEW_LINE i += 1 NEW_LINE else : NEW_LINE
m1 = m2 NEW_LINE m2 = ar2 [ j ] NEW_LINE j += 1 NEW_LINE return ( m1 + m2 ) / 2 NEW_LINE
ar1 = [ 1 , 12 , 15 , 26 , 38 ] NEW_LINE ar2 = [ 2 , 13 , 17 , 30 , 45 ] NEW_LINE n1 = len ( ar1 ) NEW_LINE n2 = len ( ar2 ) NEW_LINE if n1 == n2 : NEW_LINE INDENT print ( " Median ▁ is ▁ " , getMedian ( ar1 , ar2 , n1 ) ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Doesn ' t ▁ work ▁ for ▁ arrays ▁ of ▁ unequal ▁ size " ) NEW_LINE DEDENT
import math NEW_LINE import copy NEW_LINE
class Point ( ) : NEW_LINE INDENT def __init__ ( self , x , y ) : NEW_LINE INDENT self . x = x NEW_LINE self . y = y NEW_LINE DEDENT DEDENT
def dist ( p1 , p2 ) : NEW_LINE INDENT return math . sqrt ( ( p1 . x - p2 . x ) * ( p1 . x - p2 . x ) + ( p1 . y - p2 . y ) * ( p1 . y - p2 . y ) ) NEW_LINE DEDENT
def bruteForce ( P , n ) : NEW_LINE INDENT min_val = float ( ' inf ' ) NEW_LINE for i in range ( n ) : NEW_LINE INDENT for j in range ( i + 1 , n ) : NEW_LINE INDENT if dist ( P [ i ] , P [ j ] ) < min_val : NEW_LINE INDENT min_val = dist ( P [ i ] , P [ j ] ) NEW_LINE DEDENT DEDENT DEDENT return min_val NEW_LINE DEDENT
def stripClosest ( strip , size , d ) : NEW_LINE
min_val = d NEW_LINE
for i in range ( size ) : NEW_LINE INDENT j = i + 1 NEW_LINE while j < size and ( strip [ j ] . y - strip [ i ] . y ) < min_val : NEW_LINE INDENT min_val = dist ( strip [ i ] , strip [ j ] ) NEW_LINE j += 1 NEW_LINE DEDENT DEDENT return min_val NEW_LINE
def closestUtil ( P , Q , n ) : NEW_LINE
if n <= 3 : NEW_LINE INDENT return bruteForce ( P , n ) NEW_LINE DEDENT
mid = n // 2 NEW_LINE midPoint = P [ mid ] NEW_LINE
dl = closestUtil ( Pl , Q , mid ) NEW_LINE dr = closestUtil ( Pr , Q , n - mid ) NEW_LINE
d = min ( dl , dr ) NEW_LINE
stripP = [ ] NEW_LINE stripQ = [ ] NEW_LINE lr = Pl + Pr NEW_LINE for i in range ( n ) : NEW_LINE INDENT if abs ( lr [ i ] . x - midPoint . x ) < d : NEW_LINE INDENT stripP . append ( lr [ i ] ) NEW_LINE DEDENT if abs ( Q [ i ] . x - midPoint . x ) < d : NEW_LINE INDENT stripQ . append ( Q [ i ] ) NEW_LINE DEDENT DEDENT
return min ( min_a , min_b ) NEW_LINE
def closest ( P , n ) : NEW_LINE INDENT P . sort ( key = lambda point : point . x ) NEW_LINE Q = copy . deepcopy ( P ) NEW_LINE Q . sort ( key = lambda point : point . y ) NEW_LINE DEDENT
return closestUtil ( P , Q , n ) NEW_LINE
P = [ Point ( 2 , 3 ) , Point ( 12 , 30 ) , Point ( 40 , 50 ) , Point ( 5 , 1 ) , Point ( 12 , 10 ) , Point ( 3 , 4 ) ] NEW_LINE n = len ( P ) NEW_LINE print ( " The ▁ smallest ▁ distance ▁ is " , closest ( P , n ) ) NEW_LINE
def isLucky ( n ) : NEW_LINE
next_position = n NEW_LINE if isLucky . counter > n : NEW_LINE INDENT return 1 NEW_LINE DEDENT if n % isLucky . counter == 0 : NEW_LINE INDENT return 0 NEW_LINE DEDENT
next_position = next_position - next_position / isLucky . counter NEW_LINE isLucky . counter = isLucky . counter + 1 NEW_LINE return isLucky ( next_position ) NEW_LINE
isLucky . counter = 2 NEW_LINE x = 5 NEW_LINE if isLucky ( x ) : NEW_LINE INDENT print x , " is ▁ a ▁ Lucky ▁ number " NEW_LINE DEDENT else : NEW_LINE INDENT print x , " is ▁ not ▁ a ▁ Lucky ▁ number " NEW_LINE DEDENT
def pow ( a , b ) : NEW_LINE INDENT if ( b == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT answer = a NEW_LINE increment = a NEW_LINE for i in range ( 1 , b ) : NEW_LINE INDENT for j in range ( 1 , a ) : NEW_LINE INDENT answer += increment NEW_LINE DEDENT increment = answer NEW_LINE DEDENT return answer NEW_LINE DEDENT
print ( pow ( 5 , 3 ) ) NEW_LINE
def multiply ( x , y ) : NEW_LINE INDENT if ( y ) : NEW_LINE INDENT return ( x + multiply ( x , y - 1 ) ) ; NEW_LINE DEDENT else : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT DEDENT
def pow ( a , b ) : NEW_LINE INDENT if ( b ) : NEW_LINE INDENT return multiply ( a , pow ( a , b - 1 ) ) ; NEW_LINE DEDENT else : NEW_LINE INDENT return 1 ; NEW_LINE DEDENT DEDENT
print ( pow ( 5 , 3 ) ) ; NEW_LINE
def count ( n ) : NEW_LINE
if n < 3 : NEW_LINE INDENT return n NEW_LINE DEDENT elif n >= 3 and n < 10 : NEW_LINE INDENT return n - 1 NEW_LINE DEDENT
po = 1 NEW_LINE while n / po > 9 : NEW_LINE INDENT po = po * 10 NEW_LINE DEDENT
msd = n / po NEW_LINE if msd != 3 : NEW_LINE
return count ( msd ) * count ( po - 1 ) + count ( msd ) + count ( n % po ) NEW_LINE else : NEW_LINE
return count ( msd * po - 1 ) NEW_LINE
n = 578 NEW_LINE print count ( n ) NEW_LINE
def fact ( n ) : NEW_LINE INDENT f = 1 NEW_LINE while n >= 1 : NEW_LINE INDENT f = f * n NEW_LINE n = n - 1 NEW_LINE DEDENT return f NEW_LINE DEDENT
def findSmallerInRight ( st , low , high ) : NEW_LINE INDENT countRight = 0 NEW_LINE i = low + 1 NEW_LINE while i <= high : NEW_LINE INDENT if st [ i ] < st [ low ] : NEW_LINE INDENT countRight = countRight + 1 NEW_LINE DEDENT i = i + 1 NEW_LINE DEDENT return countRight NEW_LINE DEDENT
def findRank ( st ) : NEW_LINE INDENT ln = len ( st ) NEW_LINE mul = fact ( ln ) NEW_LINE rank = 1 NEW_LINE i = 0 NEW_LINE while i < ln : NEW_LINE INDENT mul = mul / ( ln - i ) NEW_LINE DEDENT DEDENT
countRight = findSmallerInRight ( st , i , ln - 1 ) NEW_LINE rank = rank + countRight * mul NEW_LINE i = i + 1 NEW_LINE return rank NEW_LINE
st = " string " NEW_LINE print ( findRank ( st ) ) NEW_LINE
MAX_CHAR = 256 ; NEW_LINE
count = [ 0 ] * ( MAX_CHAR + 1 ) ; NEW_LINE
def fact ( n ) : NEW_LINE INDENT return 1 if ( n <= 1 ) else ( n * fact ( n - 1 ) ) ; NEW_LINE DEDENT
def populateAndIncreaseCount ( str ) : NEW_LINE INDENT for i in range ( len ( str ) ) : NEW_LINE INDENT count [ ord ( str [ i ] ) ] += 1 ; NEW_LINE DEDENT for i in range ( 1 , MAX_CHAR ) : NEW_LINE INDENT count [ i ] += count [ i - 1 ] ; NEW_LINE DEDENT DEDENT
def updatecount ( ch ) : NEW_LINE INDENT for i in range ( ord ( ch ) , MAX_CHAR ) : NEW_LINE INDENT count [ i ] -= 1 ; NEW_LINE DEDENT DEDENT
def findRank ( str ) : NEW_LINE INDENT len1 = len ( str ) ; NEW_LINE mul = fact ( len1 ) ; NEW_LINE rank = 1 ; NEW_LINE DEDENT
populateAndIncreaseCount ( str ) ; NEW_LINE for i in range ( len1 ) : NEW_LINE INDENT mul = mul // ( len1 - i ) ; NEW_LINE DEDENT
rank += count [ ord ( str [ i ] ) - 1 ] * mul ; NEW_LINE
updatecount ( str [ i ] ) ; NEW_LINE return rank ; NEW_LINE
str = " string " ; NEW_LINE print ( findRank ( str ) ) ; NEW_LINE
def reverse ( str , l , h ) : NEW_LINE INDENT while ( l < h ) : NEW_LINE INDENT str [ l ] , str [ h ] = str [ h ] , str [ l ] NEW_LINE l += 1 NEW_LINE h -= 1 NEW_LINE DEDENT return str NEW_LINE DEDENT
def findCeil ( str , c , k , n ) : NEW_LINE INDENT ans = - 1 NEW_LINE val = c NEW_LINE for i in range ( k , n + 1 ) : NEW_LINE INDENT if str [ i ] > c and str [ i ] < val : NEW_LINE INDENT val = str [ i ] NEW_LINE ans = i NEW_LINE DEDENT DEDENT return ans NEW_LINE DEDENT
def sortedPermutations ( str ) : NEW_LINE
size = len ( str ) NEW_LINE
str = ' ' . join ( sorted ( str ) ) NEW_LINE
isFinished = False NEW_LINE while ( not isFinished ) : NEW_LINE
print ( str ) NEW_LINE
for i in range ( size - 2 , - 1 , - 1 ) : NEW_LINE INDENT if ( str [ i ] < str [ i + 1 ] ) : NEW_LINE INDENT break NEW_LINE DEDENT DEDENT
if ( i == - 1 ) : NEW_LINE INDENT isFinished = True NEW_LINE DEDENT else : NEW_LINE
ceilIndex = findCeil ( str , str [ i ] , i + 1 , size - 1 ) NEW_LINE
str [ i ] , str [ ceilIndex ] = str [ ceilIndex ] , str [ i ] NEW_LINE
str = reverse ( str , i + 1 , size - 1 ) NEW_LINE
def exponential ( n , x ) : NEW_LINE
sum = 1.0 NEW_LINE for i in range ( n , 0 , - 1 ) : NEW_LINE INDENT sum = 1 + x * sum / i NEW_LINE DEDENT print ( " e ^ x ▁ = " , sum ) NEW_LINE
n = 10 NEW_LINE x = 1.0 NEW_LINE exponential ( n , x ) NEW_LINE
import random NEW_LINE
def findCeil ( arr , r , l , h ) : NEW_LINE INDENT while ( l < h ) : NEW_LINE DEDENT
mid = l + ( ( h - l ) >> 1 ) ; NEW_LINE if r > arr [ mid ] : NEW_LINE INDENT l = mid + 1 NEW_LINE DEDENT else : NEW_LINE INDENT h = mid NEW_LINE DEDENT if arr [ l ] >= r : NEW_LINE return l NEW_LINE else : NEW_LINE return - 1 NEW_LINE
def myRand ( arr , freq , n ) : NEW_LINE
prefix = [ 0 ] * n NEW_LINE prefix [ 0 ] = freq [ 0 ] NEW_LINE for i in range ( n ) : NEW_LINE INDENT prefix [ i ] = prefix [ i - 1 ] + freq [ i ] NEW_LINE DEDENT
r = random . randint ( 0 , prefix [ n - 1 ] ) + 1 NEW_LINE
indexc = findCeil ( prefix , r , 0 , n - 1 ) NEW_LINE return arr [ indexc ] NEW_LINE
arr = [ 1 , 2 , 3 , 4 ] NEW_LINE freq = [ 10 , 5 , 20 , 100 ] NEW_LINE n = len ( arr ) NEW_LINE
for i in range ( 5 ) : NEW_LINE INDENT print ( myRand ( arr , freq , n ) ) NEW_LINE DEDENT
def random ( x , y , z , px , py , pz ) : NEW_LINE
r = random . randint ( 1 , 100 ) NEW_LINE
if ( r <= px ) : NEW_LINE INDENT return x NEW_LINE DEDENT
if ( r <= ( px + py ) ) : NEW_LINE INDENT return y NEW_LINE DEDENT
else : NEW_LINE INDENT return z NEW_LINE DEDENT
def calcAngle ( h , m ) : NEW_LINE
if ( h < 0 or m < 0 or h > 12 or m > 60 ) : NEW_LINE INDENT print ( ' Wrong ▁ input ' ) NEW_LINE DEDENT if ( h == 12 ) : NEW_LINE INDENT h = 0 NEW_LINE DEDENT if ( m == 60 ) : NEW_LINE INDENT m = 0 NEW_LINE h += 1 ; NEW_LINE if ( h > 12 ) : NEW_LINE INDENT h = h - 12 ; NEW_LINE DEDENT DEDENT
hour_angle = 0.5 * ( h * 60 + m ) NEW_LINE minute_angle = 6 * m NEW_LINE
angle = abs ( hour_angle - minute_angle ) NEW_LINE
angle = min ( 360 - angle , angle ) NEW_LINE return angle NEW_LINE
h = 9 NEW_LINE m = 60 NEW_LINE print ( ' Angle ▁ ' , calcAngle ( h , m ) ) NEW_LINE
def getSingle ( arr , n ) : NEW_LINE INDENT ones = 0 NEW_LINE twos = 0 NEW_LINE for i in range ( n ) : NEW_LINE DEDENT
twos = twos | ( ones & arr [ i ] ) NEW_LINE
ones = ones ^ arr [ i ] NEW_LINE
common_bit_mask = ~ ( ones & twos ) NEW_LINE
ones &= common_bit_mask NEW_LINE
twos &= common_bit_mask NEW_LINE return ones NEW_LINE
arr = [ 3 , 3 , 2 , 3 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " The ▁ element ▁ with ▁ single ▁ occurrence ▁ is ▁ " , getSingle ( arr , n ) ) NEW_LINE
INT_SIZE = 32 NEW_LINE def getSingle ( arr , n ) : NEW_LINE
result = 0 NEW_LINE
for i in range ( 0 , INT_SIZE ) : NEW_LINE
sm = 0 NEW_LINE x = ( 1 << i ) NEW_LINE for j in range ( 0 , n ) : NEW_LINE INDENT if ( arr [ j ] & x ) : NEW_LINE INDENT sm = sm + 1 NEW_LINE DEDENT DEDENT
if ( ( sm % 3 ) != 0 ) : NEW_LINE INDENT result = result | x NEW_LINE DEDENT return result NEW_LINE
arr = [ 12 , 1 , 12 , 3 , 12 , 1 , 1 , 2 , 3 , 2 , 2 , 3 , 7 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " The ▁ element ▁ with ▁ single ▁ occurrence ▁ is ▁ " , getSingle ( arr , n ) ) NEW_LINE
def getLeftmostBit ( n ) : NEW_LINE INDENT m = 0 NEW_LINE while ( n > 1 ) : NEW_LINE INDENT n = n >> 1 NEW_LINE m += 1 NEW_LINE DEDENT return m NEW_LINE DEDENT
def getNextLeftmostBit ( n , m ) : NEW_LINE INDENT temp = 1 << m NEW_LINE while ( n < temp ) : NEW_LINE INDENT temp = temp >> 1 NEW_LINE m -= 1 NEW_LINE DEDENT return m NEW_LINE DEDENT
def countSetBits ( n ) : NEW_LINE
m = getLeftmostBit ( n ) NEW_LINE
return _countSetBits ( n , m ) NEW_LINE def _countSetBits ( n , m ) : NEW_LINE
if ( n == 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
m = getNextLeftmostBit ( n , m ) NEW_LINE
if ( n == ( 1 << ( m + 1 ) ) - 1 ) : NEW_LINE INDENT return ( ( m + 1 ) * ( 1 << m ) ) NEW_LINE DEDENT
n = n - ( 1 << m ) NEW_LINE return ( n + 1 ) + countSetBits ( n ) + m * ( 1 << ( m - 1 ) ) NEW_LINE
n = 17 NEW_LINE print ( " Total ▁ set ▁ bit ▁ count ▁ is " , countSetBits ( n ) ) NEW_LINE
def swapBits ( x , p1 , p2 , n ) : NEW_LINE
set1 = ( x >> p1 ) & ( ( 1 << n ) - 1 ) NEW_LINE
set2 = ( x >> p2 ) & ( ( 1 << n ) - 1 ) NEW_LINE
xor = ( set1 ^ set2 ) NEW_LINE
xor = ( xor << p1 ) | ( xor << p2 ) NEW_LINE
result = x ^ xor NEW_LINE return result NEW_LINE
res = swapBits ( 28 , 0 , 3 , 2 ) NEW_LINE print ( " Result ▁ = " , res ) NEW_LINE
def smallest ( x , y , z ) : NEW_LINE INDENT c = 0 NEW_LINE while ( x and y and z ) : NEW_LINE INDENT x = x - 1 NEW_LINE y = y - 1 NEW_LINE z = z - 1 NEW_LINE c = c + 1 NEW_LINE DEDENT return c NEW_LINE DEDENT
x = 12 NEW_LINE y = 15 NEW_LINE z = 5 NEW_LINE print ( " Minimum ▁ of ▁ 3 ▁ numbers ▁ is " , smallest ( x , y , z ) ) NEW_LINE
CHAR_BIT = 8 NEW_LINE
def min ( x , y ) : NEW_LINE INDENT return y + ( ( x - y ) & ( ( x - y ) >> ( 32 * CHAR_BIT - 1 ) ) ) NEW_LINE DEDENT
def smallest ( x , y , z ) : NEW_LINE INDENT return min ( x , min ( y , z ) ) NEW_LINE DEDENT
x = 12 NEW_LINE y = 15 NEW_LINE z = 5 NEW_LINE print ( " Minimum ▁ of ▁ 3 ▁ numbers ▁ is ▁ " , smallest ( x , y , z ) ) NEW_LINE
def smallest ( x , y , z ) : NEW_LINE
if ( not ( y / x ) ) : NEW_LINE INDENT return y if ( not ( y / z ) ) else z NEW_LINE DEDENT return x if ( not ( x / z ) ) else z NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT x = 78 NEW_LINE y = 88 NEW_LINE z = 68 NEW_LINE print ( " Minimum ▁ of ▁ 3 ▁ numbers ▁ is " , smallest ( x , y , z ) ) NEW_LINE DEDENT
def changeToZero ( a ) : NEW_LINE INDENT a [ a [ 1 ] ] = a [ not a [ 1 ] ] NEW_LINE return a NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT a = [ 1 , 0 ] NEW_LINE a = changeToZero ( a ) ; NEW_LINE print ( " ▁ arr [ 0 ] ▁ = ▁ " + str ( a [ 0 ] ) ) NEW_LINE print ( " ▁ arr [ 1 ] ▁ = ▁ " + str ( a [ 1 ] ) ) NEW_LINE DEDENT
def addOne ( x ) : NEW_LINE INDENT m = 1 ; NEW_LINE DEDENT
while ( x & m ) : NEW_LINE INDENT x = x ^ m NEW_LINE m <<= 1 NEW_LINE DEDENT
x = x ^ m NEW_LINE return x NEW_LINE
n = 13 NEW_LINE print addOne ( n ) NEW_LINE
def addOne ( x ) : NEW_LINE INDENT return ( - ( ~ x ) ) ; NEW_LINE DEDENT
print ( addOne ( 13 ) ) NEW_LINE
def fun ( n ) : NEW_LINE INDENT return n & ( n - 1 ) NEW_LINE DEDENT
n = 7 NEW_LINE print ( " The ▁ number ▁ after ▁ unsetting ▁ the ▁ rightmost ▁ set ▁ bit " , fun ( n ) ) NEW_LINE
def isPowerOfFour ( n ) : NEW_LINE INDENT if ( n == 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT while ( n != 1 ) : NEW_LINE INDENT if ( n % 4 != 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT n = n // 4 NEW_LINE DEDENT return True NEW_LINE DEDENT
test_no = 64 NEW_LINE if ( isPowerOfFour ( 64 ) ) : NEW_LINE INDENT print ( test_no , ' is ▁ a ▁ power ▁ of ▁ 4' ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( test_no , ' is ▁ not ▁ a ▁ power ▁ of ▁ 4' ) NEW_LINE DEDENT
def isPowerOfFour ( n ) : NEW_LINE INDENT count = 0 NEW_LINE DEDENT
if ( n and ( not ( n & ( n - 1 ) ) ) ) : NEW_LINE
while ( n > 1 ) : NEW_LINE INDENT n >>= 1 NEW_LINE count += 1 NEW_LINE DEDENT
if ( count % 2 == 0 ) : NEW_LINE INDENT return True NEW_LINE DEDENT else : NEW_LINE INDENT return False NEW_LINE DEDENT
test_no = 64 NEW_LINE if ( isPowerOfFour ( 64 ) ) : NEW_LINE INDENT print ( test_no , ' is ▁ a ▁ power ▁ of ▁ 4' ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( test_no , ' is ▁ not ▁ a ▁ power ▁ of ▁ 4' ) NEW_LINE DEDENT
def isPowerOfFour ( n ) : NEW_LINE INDENT return ( n != 0 and ( ( n & ( n - 1 ) ) == 0 ) and not ( n & 0xAAAAAAAA ) ) ; NEW_LINE DEDENT
test_no = 64 ; NEW_LINE if ( isPowerOfFour ( test_no ) ) : NEW_LINE INDENT print ( test_no , " is ▁ a ▁ power ▁ of ▁ 4" ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( test_no , " is ▁ not ▁ a ▁ power ▁ of ▁ 4" ) ; NEW_LINE DEDENT
def min ( x , y ) : NEW_LINE INDENT return y ^ ( ( x ^ y ) & - ( x < y ) ) NEW_LINE DEDENT
def max ( x , y ) : NEW_LINE INDENT return x ^ ( ( x ^ y ) & - ( x < y ) ) NEW_LINE DEDENT
x = 15 NEW_LINE y = 6 NEW_LINE print ( " Minimum ▁ of " , x , " and " , y , " is " , end = " ▁ " ) NEW_LINE print ( min ( x , y ) ) NEW_LINE print ( " Maximum ▁ of " , x , " and " , y , " is " , end = " ▁ " ) NEW_LINE print ( max ( x , y ) ) NEW_LINE
import sys ; NEW_LINE CHAR_BIT = 8 ; NEW_LINE INT_BIT = sys . getsizeof ( int ( ) ) ; NEW_LINE
def Min ( x , y ) : NEW_LINE INDENT return y + ( ( x - y ) & ( ( x - y ) >> ( INT_BIT * CHAR_BIT - 1 ) ) ) ; NEW_LINE DEDENT
def Max ( x , y ) : NEW_LINE INDENT return x - ( ( x - y ) & ( ( x - y ) >> ( INT_BIT * CHAR_BIT - 1 ) ) ) ; NEW_LINE DEDENT
x = 15 ; NEW_LINE y = 6 ; NEW_LINE print ( " Minimum ▁ of " , x , " and " , y , " is " , Min ( x , y ) ) ; NEW_LINE print ( " Maximum ▁ of " , x , " and " , y , " is " , Max ( x , y ) ) ; NEW_LINE
import math NEW_LINE def getFirstSetBitPos ( n ) : NEW_LINE INDENT return math . log2 ( n & - n ) + 1 NEW_LINE DEDENT
n = 12 NEW_LINE print ( int ( getFirstSetBitPos ( n ) ) ) NEW_LINE
def bin ( n ) : NEW_LINE INDENT i = 1 << 31 NEW_LINE while ( i > 0 ) : NEW_LINE INDENT if ( ( n & i ) != 0 ) : NEW_LINE INDENT print ( "1" , end = " " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( "0" , end = " " ) NEW_LINE DEDENT i = i // 2 NEW_LINE DEDENT DEDENT
bin ( 7 ) NEW_LINE print ( ) NEW_LINE bin ( 4 ) NEW_LINE
def swapBits ( x ) : NEW_LINE
even_bits = x & 0xAAAAAAAA NEW_LINE
odd_bits = x & 0x55555555 NEW_LINE
even_bits >>= 1 NEW_LINE
odd_bits <<= 1 NEW_LINE
return ( even_bits odd_bits ) NEW_LINE
x = 23 NEW_LINE
print ( swapBits ( x ) ) NEW_LINE
def isPowerOfTwo ( n ) : NEW_LINE INDENT return ( True if ( n > 0 and ( ( n & ( n - 1 ) ) > 0 ) ) else False ) ; NEW_LINE DEDENT
def findPosition ( n ) : NEW_LINE INDENT if ( isPowerOfTwo ( n ) == True ) : NEW_LINE INDENT return - 1 ; NEW_LINE DEDENT i = 1 ; NEW_LINE pos = 1 ; NEW_LINE DEDENT
while ( ( i & n ) == 0 ) : NEW_LINE
i = i << 1 ; NEW_LINE
pos += 1 ; NEW_LINE return pos ; NEW_LINE
n = 16 ; NEW_LINE pos = findPosition ( n ) ; NEW_LINE if ( pos == - 1 ) : NEW_LINE INDENT print ( " n ▁ = " , n , " , ▁ Invalid ▁ number " ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " n ▁ = " , n , " , ▁ Position ▁ " , pos ) ; NEW_LINE DEDENT n = 12 ; NEW_LINE pos = findPosition ( n ) ; NEW_LINE if ( pos == - 1 ) : NEW_LINE INDENT print ( " n ▁ = " , n , " , ▁ Invalid ▁ number " ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " n ▁ = " , n , " , ▁ Position ▁ " , pos ) ; NEW_LINE DEDENT n = 128 ; NEW_LINE pos = findPosition ( n ) ; NEW_LINE if ( pos == - 1 ) : NEW_LINE INDENT print ( " n ▁ = " , n , " , ▁ Invalid ▁ number " ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " n ▁ = " , n , " , ▁ Position ▁ " , pos ) ; NEW_LINE DEDENT
def isPowerOfTwo ( n ) : NEW_LINE INDENT return ( n and ( not ( n & ( n - 1 ) ) ) ) NEW_LINE DEDENT
def findPosition ( n ) : NEW_LINE INDENT if not isPowerOfTwo ( n ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT count = 0 NEW_LINE DEDENT
while ( n ) : NEW_LINE INDENT n = n >> 1 NEW_LINE DEDENT
count += 1 NEW_LINE return count NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 0 NEW_LINE pos = findPosition ( n ) NEW_LINE if pos == - 1 : NEW_LINE INDENT print ( " n ▁ = " , n , " Invalid ▁ number " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " n ▁ = " , n , " Position " , pos ) NEW_LINE DEDENT n = 12 NEW_LINE pos = findPosition ( n ) NEW_LINE if pos == - 1 : NEW_LINE INDENT print ( " n ▁ = " , n , " Invalid ▁ number " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " n ▁ = " , n , " Position " , pos ) NEW_LINE DEDENT n = 128 NEW_LINE pos = findPosition ( n ) NEW_LINE if pos == - 1 : NEW_LINE INDENT print ( " n ▁ = " , n , " Invalid ▁ number " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " n ▁ = " , n , " Position " , pos ) NEW_LINE DEDENT DEDENT
x = 10 NEW_LINE y = 5 NEW_LINE
x = x * y NEW_LINE
y = x // y ; NEW_LINE
x = x // y ; NEW_LINE print ( " After ▁ Swapping : ▁ x ▁ = " , x , " ▁ y ▁ = " , y ) ; NEW_LINE
x = 10 NEW_LINE y = 5 NEW_LINE
x = x ^ y ; NEW_LINE
y = x ^ y ; NEW_LINE
x = x ^ y ; NEW_LINE print ( " After ▁ Swapping : ▁ x ▁ = ▁ " , x , " ▁ y ▁ = " , y ) NEW_LINE
def swap ( xp , yp ) : NEW_LINE INDENT xp [ 0 ] = xp [ 0 ] ^ yp [ 0 ] NEW_LINE yp [ 0 ] = xp [ 0 ] ^ yp [ 0 ] NEW_LINE xp [ 0 ] = xp [ 0 ] ^ yp [ 0 ] NEW_LINE DEDENT
x = [ 10 ] NEW_LINE swap ( x , x ) NEW_LINE print ( " After ▁ swap ( & x , ▁ & x ) : ▁ x ▁ = ▁ " , x [ 0 ] ) NEW_LINE
def nextGreatest ( arr ) : NEW_LINE INDENT size = len ( arr ) NEW_LINE DEDENT
max_from_right = arr [ size - 1 ] NEW_LINE
arr [ size - 1 ] = - 1 NEW_LINE
for i in range ( size - 2 , - 1 , - 1 ) : NEW_LINE
temp = arr [ i ] NEW_LINE
arr [ i ] = max_from_right NEW_LINE
if max_from_right < temp : NEW_LINE INDENT max_from_right = temp NEW_LINE DEDENT
def printArray ( arr ) : NEW_LINE INDENT for i in range ( 0 , len ( arr ) ) : NEW_LINE INDENT print arr [ i ] , NEW_LINE DEDENT DEDENT
arr = [ 16 , 17 , 4 , 3 , 5 , 2 ] NEW_LINE nextGreatest ( arr ) NEW_LINE print " Modified ▁ array ▁ is " NEW_LINE printArray ( arr ) NEW_LINE
def maxDiff ( arr , arr_size ) : NEW_LINE INDENT max_diff = arr [ 1 ] - arr [ 0 ] NEW_LINE for i in range ( 0 , arr_size ) : NEW_LINE INDENT for j in range ( i + 1 , arr_size ) : NEW_LINE INDENT if ( arr [ j ] - arr [ i ] > max_diff ) : NEW_LINE INDENT max_diff = arr [ j ] - arr [ i ] NEW_LINE DEDENT DEDENT DEDENT return max_diff NEW_LINE DEDENT
arr = [ 1 , 2 , 90 , 10 , 110 ] NEW_LINE size = len ( arr ) NEW_LINE
print ( " Maximum ▁ difference ▁ is " , maxDiff ( arr , size ) ) NEW_LINE
def findMaximum ( arr , low , high ) : NEW_LINE INDENT max = arr [ low ] NEW_LINE i = low NEW_LINE for i in range ( high + 1 ) : NEW_LINE INDENT if arr [ i ] > max : NEW_LINE INDENT max = arr [ i ] NEW_LINE DEDENT DEDENT return max NEW_LINE DEDENT
arr = [ 1 , 30 , 40 , 50 , 60 , 70 , 23 , 20 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " The ▁ maximum ▁ element ▁ is ▁ % d " % findMaximum ( arr , 0 , n - 1 ) ) NEW_LINE
def findMaximum ( arr , low , high ) : NEW_LINE
if low == high : NEW_LINE INDENT return arr [ low ] NEW_LINE DEDENT
if high == low + 1 and arr [ low ] >= arr [ high ] : NEW_LINE INDENT return arr [ low ] ; NEW_LINE DEDENT
if high == low + 1 and arr [ low ] < arr [ high ] : NEW_LINE INDENT return arr [ high ] NEW_LINE DEDENT mid = ( low + high ) // 2 NEW_LINE
if arr [ mid ] > arr [ mid + 1 ] and arr [ mid ] > arr [ mid - 1 ] : NEW_LINE INDENT return arr [ mid ] NEW_LINE DEDENT
if arr [ mid ] > arr [ mid + 1 ] and arr [ mid ] < arr [ mid - 1 ] : NEW_LINE INDENT return findMaximum ( arr , low , mid - 1 ) NEW_LINE DEDENT
else : NEW_LINE INDENT return findMaximum ( arr , mid + 1 , high ) NEW_LINE DEDENT
arr = [ 1 , 3 , 50 , 10 , 9 , 7 , 6 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " The ▁ maximum ▁ element ▁ is ▁ % d " % findMaximum ( arr , 0 , n - 1 ) ) NEW_LINE
def constructLowerArray ( arr , countSmaller , n ) : NEW_LINE
for i in range ( n ) : NEW_LINE INDENT countSmaller [ i ] = 0 NEW_LINE DEDENT for i in range ( n ) : NEW_LINE INDENT for j in range ( i + 1 , n ) : NEW_LINE INDENT if ( arr [ j ] < arr [ i ] ) : NEW_LINE INDENT countSmaller [ i ] += 1 NEW_LINE DEDENT DEDENT DEDENT
def printArray ( arr , size ) : NEW_LINE INDENT for i in range ( size ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT print ( ) NEW_LINE DEDENT
arr = [ 12 , 10 , 5 , 4 , 2 , 20 , 6 , 1 , 0 , 2 ] NEW_LINE n = len ( arr ) NEW_LINE low = [ 0 ] * n NEW_LINE constructLowerArray ( arr , low , n ) NEW_LINE printArray ( low , n ) NEW_LINE
def segregate ( arr , size ) : NEW_LINE INDENT j = 0 NEW_LINE for i in range ( size ) : NEW_LINE INDENT if ( arr [ i ] <= 0 ) : NEW_LINE INDENT arr [ i ] , arr [ j ] = arr [ j ] , arr [ i ] NEW_LINE DEDENT DEDENT DEDENT
j += 1 NEW_LINE return j NEW_LINE
def findMissingPositive ( arr , size ) : NEW_LINE
for i in range ( size ) : NEW_LINE INDENT if ( abs ( arr [ i ] ) - 1 < size and arr [ abs ( arr [ i ] ) - 1 ] > 0 ) : NEW_LINE INDENT arr [ abs ( arr [ i ] ) - 1 ] = - arr [ abs ( arr [ i ] ) - 1 ] NEW_LINE DEDENT DEDENT
for i in range ( size ) : NEW_LINE INDENT if ( arr [ i ] > 0 ) : NEW_LINE DEDENT
return i + 1 NEW_LINE return size + 1 NEW_LINE
def findMissing ( arr , size ) : NEW_LINE
shift = segregate ( arr , size ) NEW_LINE
return findMissingPositive ( arr [ shift : ] , size - shift ) NEW_LINE
arr = [ 0 , 10 , 2 , - 10 , - 20 ] NEW_LINE arr_size = len ( arr ) NEW_LINE missing = findMissing ( arr , arr_size ) NEW_LINE print ( " The ▁ smallest ▁ positive ▁ missing ▁ number ▁ is ▁ " , missing ) NEW_LINE
def getMissingNo ( A ) : NEW_LINE INDENT n = len ( A ) NEW_LINE total = ( n + 1 ) * ( n + 2 ) / 2 NEW_LINE sum_of_A = sum ( A ) NEW_LINE return total - sum_of_A NEW_LINE DEDENT
A = [ 1 , 2 , 4 , 5 , 6 ] NEW_LINE miss = getMissingNo ( A ) NEW_LINE print ( miss ) NEW_LINE
def printTwoElements ( arr , size ) : NEW_LINE INDENT for i in range ( size ) : NEW_LINE INDENT if arr [ abs ( arr [ i ] ) - 1 ] > 0 : NEW_LINE INDENT arr [ abs ( arr [ i ] ) - 1 ] = - arr [ abs ( arr [ i ] ) - 1 ] NEW_LINE DEDENT else : NEW_LINE INDENT print ( " The ▁ repeating ▁ element ▁ is " , abs ( arr [ i ] ) ) NEW_LINE DEDENT DEDENT for i in range ( size ) : NEW_LINE INDENT if arr [ i ] > 0 : NEW_LINE INDENT print ( " and ▁ the ▁ missing ▁ element ▁ is " , i + 1 ) NEW_LINE DEDENT DEDENT DEDENT
arr = [ 7 , 3 , 4 , 5 , 5 , 6 , 2 ] NEW_LINE n = len ( arr ) NEW_LINE printTwoElements ( arr , n ) NEW_LINE
def getTwoElements ( arr , n ) : NEW_LINE INDENT global x , y NEW_LINE x = 0 NEW_LINE y = 0 NEW_LINE DEDENT
xor1 = arr [ 0 ] NEW_LINE
for i in range ( 1 , n ) : NEW_LINE INDENT xor1 = xor1 ^ arr [ i ] NEW_LINE DEDENT
for i in range ( 1 , n + 1 ) : NEW_LINE INDENT xor1 = xor1 ^ i NEW_LINE DEDENT
set_bit_no = xor1 & ~ ( xor1 - 1 ) NEW_LINE
for i in range ( n ) : NEW_LINE INDENT if ( arr [ i ] & set_bit_no ) != 0 : NEW_LINE DEDENT
x = x ^ arr [ i ] NEW_LINE else : NEW_LINE
y = y ^ arr [ i ] NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE if ( i & set_bit_no ) != 0 : NEW_LINE
x = x ^ i NEW_LINE else : NEW_LINE
y = y ^ i NEW_LINE
arr = [ 1 , 3 , 4 , 5 , 5 , 6 , 2 ] NEW_LINE n = len ( arr ) NEW_LINE getTwoElements ( arr , n ) NEW_LINE print ( " The ▁ missing ▁ element ▁ is " , x , " and ▁ the ▁ repeating ▁ number ▁ is " , y ) NEW_LINE
def findFourElements ( A , n , X ) : NEW_LINE
for i in range ( 0 , n - 3 ) : NEW_LINE
for j in range ( i + 1 , n - 2 ) : NEW_LINE
for k in range ( j + 1 , n - 1 ) : NEW_LINE
for l in range ( k + 1 , n ) : NEW_LINE INDENT if A [ i ] + A [ j ] + A [ k ] + A [ l ] == X : NEW_LINE INDENT print ( " % d , ▁ % d , ▁ % d , ▁ % d " % ( A [ i ] , A [ j ] , A [ k ] , A [ l ] ) ) NEW_LINE DEDENT DEDENT
A = [ 10 , 2 , 3 , 4 , 5 , 9 , 7 , 8 ] NEW_LINE n = len ( A ) NEW_LINE X = 23 NEW_LINE findFourElements ( A , n , X ) NEW_LINE
def minDistance ( arr , n ) : NEW_LINE INDENT maximum_element = arr [ 0 ] NEW_LINE min_dis = n NEW_LINE index = 0 NEW_LINE for i in range ( 1 , n ) : NEW_LINE DEDENT
if ( maximum_element == arr [ i ] ) : NEW_LINE INDENT min_dis = min ( min_dis , ( i - index ) ) NEW_LINE index = i NEW_LINE DEDENT
elif ( maximum_element < arr [ i ] ) : NEW_LINE INDENT maximum_element = arr [ i ] NEW_LINE min_dis = n NEW_LINE index = i NEW_LINE DEDENT
else : NEW_LINE INDENT continue NEW_LINE DEDENT return min_dis NEW_LINE
arr = [ 6 , 3 , 1 , 3 , 6 , 4 , 6 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Minimum ▁ distance ▁ = " , minDistance ( arr , n ) ) NEW_LINE
def deleteAlt ( head ) : NEW_LINE INDENT if ( head == None ) : NEW_LINE INDENT return NEW_LINE DEDENT node = head . next NEW_LINE if ( node == None ) : NEW_LINE INDENT return NEW_LINE DEDENT DEDENT
head . next = node . next NEW_LINE
deleteAlt ( head . next ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , d ) : NEW_LINE INDENT self . data = d NEW_LINE self . next = None NEW_LINE DEDENT DEDENT
def areIdentical ( self , listb ) : NEW_LINE INDENT a = self . head NEW_LINE b = listb . head NEW_LINE while ( a != None and b != None ) : NEW_LINE INDENT if ( a . data != b . data ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT DEDENT
a = a . next NEW_LINE b = b . next NEW_LINE
return ( a == None and b == None ) NEW_LINE
def push ( self , new_data ) : NEW_LINE
new_node = Node ( new_data ) NEW_LINE
new_node . next = self . head NEW_LINE
self . head = new_node NEW_LINE
llist1 = LinkedList ( ) NEW_LINE llist2 = LinkedList ( ) NEW_LINE
llist1 . push ( 1 ) NEW_LINE llist1 . push ( 2 ) NEW_LINE llist1 . push ( 3 ) NEW_LINE llist2 . push ( 1 ) NEW_LINE llist2 . push ( 2 ) NEW_LINE llist2 . push ( 3 ) NEW_LINE if ( llist1 . areIdentical ( llist2 ) == True ) : NEW_LINE INDENT print ( " Identical ▁ " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Not ▁ identical ▁ " ) NEW_LINE DEDENT
def areIdentical ( a , b ) : NEW_LINE
if ( a == None and b == None ) : NEW_LINE INDENT return True NEW_LINE DEDENT
if ( a != None and b != None ) : NEW_LINE INDENT return ( ( a . data == b . data ) and areIdentical ( a . next , b . next ) ) NEW_LINE DEDENT
return False NEW_LINE
class LinkedList ( object ) : NEW_LINE INDENT def __init__ ( self ) : NEW_LINE DEDENT
class Node ( object ) : NEW_LINE INDENT def __init__ ( self , d ) : NEW_LINE INDENT self . data = d NEW_LINE self . next = None NEW_LINE DEDENT DEDENT def sortList ( self ) : NEW_LINE
count = [ 0 , 0 , 0 ] NEW_LINE ptr = self . head NEW_LINE
while ptr != None : NEW_LINE INDENT count [ ptr . data ] += 1 NEW_LINE ptr = ptr . next NEW_LINE DEDENT i = 0 NEW_LINE ptr = self . head NEW_LINE
while ptr != None : NEW_LINE INDENT if count [ i ] == 0 : NEW_LINE INDENT i += 1 NEW_LINE DEDENT else : NEW_LINE INDENT ptr . data = i NEW_LINE count [ i ] -= 1 NEW_LINE ptr = ptr . next NEW_LINE DEDENT DEDENT
def push ( self , new_data ) : NEW_LINE
new_node = self . Node ( new_data ) NEW_LINE
new_node . next = self . head NEW_LINE
self . head = new_node NEW_LINE
def printList ( self ) : NEW_LINE INDENT temp = self . head NEW_LINE while temp != None : NEW_LINE INDENT print str ( temp . data ) , NEW_LINE temp = temp . next NEW_LINE DEDENT print ' ' NEW_LINE DEDENT
llist = LinkedList ( ) NEW_LINE llist . push ( 0 ) NEW_LINE llist . push ( 1 ) NEW_LINE llist . push ( 0 ) NEW_LINE llist . push ( 2 ) NEW_LINE llist . push ( 1 ) NEW_LINE llist . push ( 1 ) NEW_LINE llist . push ( 2 ) NEW_LINE llist . push ( 1 ) NEW_LINE llist . push ( 2 ) NEW_LINE print " Linked ▁ List ▁ before ▁ sorting " NEW_LINE llist . printList ( ) NEW_LINE llist . sortList ( ) NEW_LINE print " Linked ▁ List ▁ after ▁ sorting " NEW_LINE llist . printList ( ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . next = None NEW_LINE self . child = None NEW_LINE DEDENT DEDENT
class Node : NEW_LINE INDENT def __init__ ( self , d ) : NEW_LINE INDENT self . data = d NEW_LINE self . next = None NEW_LINE DEDENT DEDENT class LinkedList : NEW_LINE INDENT def __init__ ( self ) : NEW_LINE INDENT self . head = None NEW_LINE DEDENT DEDENT
def newNode ( self , key ) : NEW_LINE INDENT temp = Node ( key ) NEW_LINE self . next = None NEW_LINE return temp NEW_LINE DEDENT
def rearrangeEvenOdd ( self , head ) : NEW_LINE
if ( self . head == None ) : NEW_LINE INDENT return None NEW_LINE DEDENT
odd = self . head NEW_LINE even = self . head . next NEW_LINE
evenFirst = even NEW_LINE while ( 1 == 1 ) : NEW_LINE
if ( odd == None or even == None or ( even . next ) == None ) : NEW_LINE INDENT odd . next = evenFirst NEW_LINE break NEW_LINE DEDENT
odd . next = even . next NEW_LINE odd = even . next NEW_LINE
if ( odd . next == None ) : NEW_LINE INDENT even . next = None NEW_LINE odd . next = evenFirst NEW_LINE break NEW_LINE DEDENT
even . next = odd . next NEW_LINE even = odd . next NEW_LINE return head NEW_LINE
def printlist ( self , node ) : NEW_LINE INDENT while ( node != None ) : NEW_LINE INDENT print ( node . data , end = " " ) NEW_LINE print ( " - > " , end = " " ) NEW_LINE node = node . next NEW_LINE DEDENT print ( " NULL " ) NEW_LINE DEDENT
ll = LinkedList ( ) NEW_LINE ll . push ( 5 ) NEW_LINE ll . push ( 4 ) NEW_LINE ll . push ( 3 ) NEW_LINE ll . push ( 2 ) NEW_LINE ll . push ( 1 ) NEW_LINE print ( " Given ▁ Linked ▁ List " ) NEW_LINE ll . printlist ( ll . head ) NEW_LINE start = ll . rearrangeEvenOdd ( ll . head ) NEW_LINE print ( " Modified Linked List " ) NEW_LINE ll . printlist ( start ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , new_data ) : NEW_LINE INDENT self . data = new_data NEW_LINE self . next = None NEW_LINE DEDENT DEDENT
def deleteLast ( head , x ) : NEW_LINE INDENT temp = head NEW_LINE ptr = None NEW_LINE while ( temp != None ) : NEW_LINE DEDENT
if ( temp . data == x ) : NEW_LINE INDENT ptr = temp NEW_LINE DEDENT temp = temp . next NEW_LINE
if ( ptr != None and ptr . next == None ) : NEW_LINE INDENT temp = head NEW_LINE while ( temp . next != ptr ) : NEW_LINE INDENT temp = temp . next NEW_LINE DEDENT temp . next = None NEW_LINE DEDENT
if ( ptr != None and ptr . next != None ) : NEW_LINE INDENT ptr . data = ptr . next . data NEW_LINE temp = ptr . next NEW_LINE ptr . next = ptr . next . next NEW_LINE DEDENT return head NEW_LINE
def newNode ( x ) : NEW_LINE INDENT node = Node ( 0 ) NEW_LINE node . data = x NEW_LINE node . next = None NEW_LINE return node NEW_LINE DEDENT
def display ( head ) : NEW_LINE INDENT temp = head NEW_LINE if ( head == None ) : NEW_LINE INDENT print ( " None " ) NEW_LINE return NEW_LINE DEDENT while ( temp != None ) : NEW_LINE INDENT print ( temp . data , " ▁ - > ▁ " , end = " " ) NEW_LINE temp = temp . next NEW_LINE DEDENT print ( " None " ) NEW_LINE DEDENT
head = newNode ( 1 ) NEW_LINE head . next = newNode ( 2 ) NEW_LINE head . next . next = newNode ( 3 ) NEW_LINE head . next . next . next = newNode ( 4 ) NEW_LINE head . next . next . next . next = newNode ( 5 ) NEW_LINE head . next . next . next . next . next = newNode ( 4 ) NEW_LINE head . next . next . next . next . next . next = newNode ( 4 ) NEW_LINE print ( " Created ▁ Linked ▁ list : ▁ " ) NEW_LINE display ( head ) NEW_LINE head = deleteLast ( head , 4 ) NEW_LINE print ( " List ▁ after ▁ deletion ▁ of ▁ 4 : ▁ " ) NEW_LINE display ( head ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , d ) : NEW_LINE INDENT self . data = d NEW_LINE self . next = None NEW_LINE self . head = None NEW_LINE DEDENT DEDENT
def LinkedListLength ( self ) : NEW_LINE INDENT while ( self . head != None and self . head . next != None ) : NEW_LINE INDENT self . head = self . head . next . next NEW_LINE DEDENT if ( self . head == None ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT return 1 NEW_LINE DEDENT
def push ( self , info ) : NEW_LINE
node = Node ( info ) NEW_LINE
node . next = ( self . head ) NEW_LINE
( self . head ) = node NEW_LINE
head = Node ( 0 ) NEW_LINE
head . push ( 4 ) NEW_LINE head . push ( 5 ) NEW_LINE head . push ( 7 ) NEW_LINE head . push ( 2 ) NEW_LINE head . push ( 9 ) NEW_LINE head . push ( 6 ) NEW_LINE head . push ( 1 ) NEW_LINE head . push ( 2 ) NEW_LINE head . push ( 0 ) NEW_LINE head . push ( 5 ) NEW_LINE head . push ( 5 ) NEW_LINE check = head . LinkedListLength ( ) NEW_LINE
if ( check == 0 ) : NEW_LINE INDENT print ( " Even " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Odd " ) NEW_LINE DEDENT
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . next = None NEW_LINE DEDENT DEDENT
def setMiddleHead ( head ) : NEW_LINE INDENT if ( head == None ) : NEW_LINE INDENT return None NEW_LINE DEDENT DEDENT
one_node = head NEW_LINE
two_node = head NEW_LINE
prev = None NEW_LINE while ( two_node != None and two_node . next != None ) : NEW_LINE
prev = one_node NEW_LINE
one_node = one_node . next NEW_LINE
two_node = two_node . next . next NEW_LINE
prev . next = prev . next . next NEW_LINE one_node . next = head NEW_LINE head = one_node NEW_LINE return head NEW_LINE
def push ( head , new_data ) : NEW_LINE
new_node = Node ( new_data ) NEW_LINE
new_node . next = head NEW_LINE
head = new_node NEW_LINE return head NEW_LINE
def printList ( head ) : NEW_LINE INDENT temp = head NEW_LINE while ( temp != None ) : NEW_LINE INDENT print ( str ( temp . data ) , end = " ▁ " ) NEW_LINE temp = temp . next NEW_LINE DEDENT print ( " " ) NEW_LINE DEDENT
head = None NEW_LINE for i in range ( 5 , 0 , - 1 ) : NEW_LINE INDENT head = push ( head , i ) NEW_LINE DEDENT print ( " ▁ list ▁ before : ▁ " , end = " " ) NEW_LINE printList ( head ) NEW_LINE head = setMiddleHead ( head ) NEW_LINE print ( " ▁ list ▁ After : ▁ " , end = " " ) NEW_LINE printList ( head ) NEW_LINE
def insertAfter ( self , prev_node , new_data ) : NEW_LINE
if prev_node is None : NEW_LINE INDENT print ( " This ▁ node ▁ doesn ' t ▁ exist ▁ in ▁ DLL " ) NEW_LINE return NEW_LINE DEDENT
new_node = Node ( data = new_data ) NEW_LINE
new_node . next = prev_node . next NEW_LINE
prev_node . next = new_node NEW_LINE
new_node . prev = prev_node NEW_LINE
if new_node . next is not None : NEW_LINE INDENT new_node . next . prev = new_node NEW_LINE DEDENT
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT def printKDistant ( root , k ) : NEW_LINE INDENT if root is None : NEW_LINE INDENT return NEW_LINE DEDENT if k == 0 : NEW_LINE INDENT print root . data , NEW_LINE DEDENT else : NEW_LINE INDENT printKDistant ( root . left , k - 1 ) NEW_LINE printKDistant ( root . right , k - 1 ) NEW_LINE DEDENT DEDENT
root = Node ( 1 ) NEW_LINE root . left = Node ( 2 ) NEW_LINE root . right = Node ( 3 ) NEW_LINE root . left . left = Node ( 4 ) NEW_LINE root . left . right = Node ( 5 ) NEW_LINE root . right . left = Node ( 8 ) NEW_LINE printKDistant ( root , 2 ) NEW_LINE
COUNT = [ 10 ] NEW_LINE
class newNode : NEW_LINE
def __init__ ( self , key ) : NEW_LINE INDENT self . data = key NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT
def print2DUtil ( root , space ) : NEW_LINE
if ( root == None ) : NEW_LINE INDENT return NEW_LINE DEDENT
space += COUNT [ 0 ] NEW_LINE
print2DUtil ( root . right , space ) NEW_LINE
print ( ) NEW_LINE for i in range ( COUNT [ 0 ] , space ) : NEW_LINE INDENT print ( end = " ▁ " ) NEW_LINE DEDENT print ( root . data ) NEW_LINE
print2DUtil ( root . left , space ) NEW_LINE
def print2D ( root ) : NEW_LINE
print2DUtil ( root , 0 ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT root = newNode ( 1 ) NEW_LINE root . left = newNode ( 2 ) NEW_LINE root . right = newNode ( 3 ) NEW_LINE root . left . left = newNode ( 4 ) NEW_LINE root . left . right = newNode ( 5 ) NEW_LINE root . right . left = newNode ( 6 ) NEW_LINE root . right . right = newNode ( 7 ) NEW_LINE root . left . left . left = newNode ( 8 ) NEW_LINE root . left . left . right = newNode ( 9 ) NEW_LINE root . left . right . left = newNode ( 10 ) NEW_LINE root . left . right . right = newNode ( 11 ) NEW_LINE root . right . left . left = newNode ( 12 ) NEW_LINE root . right . left . right = newNode ( 13 ) NEW_LINE root . right . right . left = newNode ( 14 ) NEW_LINE root . right . right . right = newNode ( 15 ) NEW_LINE print2D ( root ) NEW_LINE DEDENT
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def leftViewUtil ( root , level , max_level ) : NEW_LINE
if root is None : NEW_LINE INDENT return NEW_LINE DEDENT
if ( max_level [ 0 ] < level ) : NEW_LINE INDENT print " % ▁ d TABSYMBOL " % ( root . data ) , NEW_LINE max_level [ 0 ] = level NEW_LINE DEDENT
leftViewUtil ( root . left , level + 1 , max_level ) NEW_LINE leftViewUtil ( root . right , level + 1 , max_level ) NEW_LINE
def leftView ( root ) : NEW_LINE INDENT max_level = [ 0 ] NEW_LINE leftViewUtil ( root , 1 , max_level ) NEW_LINE DEDENT
root = Node ( 12 ) NEW_LINE root . left = Node ( 10 ) NEW_LINE root . right = Node ( 20 ) NEW_LINE root . right . left = Node ( 25 ) NEW_LINE root . right . right = Node ( 40 ) NEW_LINE leftView ( root ) NEW_LINE
def cntRotations ( s , n ) : NEW_LINE INDENT lh , rh , ans = 0 , 0 , 0 NEW_LINE DEDENT
for i in range ( n // 2 ) : NEW_LINE INDENT if ( s [ i ] == ' a ' or s [ i ] == ' e ' or s [ i ] == ' i ' or s [ i ] == ' o ' or s [ i ] == ' u ' ) : NEW_LINE INDENT lh += 1 NEW_LINE DEDENT DEDENT
for i in range ( n // 2 , n ) : NEW_LINE INDENT if ( s [ i ] == ' a ' or s [ i ] == ' e ' or s [ i ] == ' i ' or s [ i ] == ' o ' or s [ i ] == ' u ' ) : NEW_LINE INDENT rh += 1 NEW_LINE DEDENT DEDENT
if ( lh > rh ) : NEW_LINE INDENT ans += 1 NEW_LINE DEDENT
for i in range ( 1 , n ) : NEW_LINE INDENT if ( s [ i - 1 ] == ' a ' or s [ i - 1 ] == ' e ' or s [ i - 1 ] == ' i ' or s [ i - 1 ] == ' o ' or s [ i - 1 ] == ' u ' ) : NEW_LINE INDENT rh += 1 NEW_LINE lh -= 1 NEW_LINE DEDENT if ( s [ ( i - 1 + n // 2 ) % n ] == ' a ' or s [ ( i - 1 + n // 2 ) % n ] == ' e ' or s [ ( i - 1 + n // 2 ) % n ] == ' i ' or s [ ( i - 1 + n // 2 ) % n ] == ' o ' or s [ ( i - 1 + n // 2 ) % n ] == ' u ' ) : NEW_LINE INDENT rh -= 1 NEW_LINE lh += 1 NEW_LINE DEDENT if ( lh > rh ) : NEW_LINE INDENT ans += 1 NEW_LINE DEDENT DEDENT
return ans NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT s = " abecidft " NEW_LINE n = len ( s ) NEW_LINE DEDENT
print ( cntRotations ( s , n ) ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . next = None NEW_LINE DEDENT DEDENT
def rotateHelper ( blockHead , blockTail , d , tail , k ) : NEW_LINE INDENT if ( d == 0 ) : NEW_LINE INDENT return blockHead , tail NEW_LINE DEDENT DEDENT
if ( d > 0 ) : NEW_LINE INDENT temp = blockHead NEW_LINE i = 1 NEW_LINE while temp . next . next != None and i < k - 1 : NEW_LINE INDENT temp = temp . next NEW_LINE i += 1 NEW_LINE DEDENT blockTail . next = blockHead NEW_LINE tail = temp NEW_LINE return rotateHelper ( blockTail , temp , d - 1 , tail , k ) NEW_LINE DEDENT
if ( d < 0 ) : NEW_LINE INDENT blockTail . next = blockHead NEW_LINE tail = blockHead NEW_LINE return rotateHelper ( blockHead . next , blockHead , d + 1 , tail , k ) NEW_LINE DEDENT
def rotateByBlocks ( head , k , d ) : NEW_LINE
if ( head == None or head . next == None ) : NEW_LINE INDENT return head NEW_LINE DEDENT
if ( d == 0 ) : NEW_LINE INDENT return head NEW_LINE DEDENT temp = head NEW_LINE tail = None NEW_LINE
i = 1 NEW_LINE while temp . next != None and i < k : NEW_LINE INDENT temp = temp . next NEW_LINE i += 1 NEW_LINE DEDENT
nextBlock = temp . next NEW_LINE
if ( i < k ) : NEW_LINE INDENT head , tail = rotateHelper ( head , temp , d % k , tail , i ) NEW_LINE DEDENT else : NEW_LINE INDENT head , tail = rotateHelper ( head , temp , d % k , tail , k ) NEW_LINE DEDENT
tail . next = rotateByBlocks ( nextBlock , k , d % k ) ; NEW_LINE
return head ; NEW_LINE
def push ( head_ref , new_data ) : NEW_LINE INDENT new_node = Node ( new_data ) NEW_LINE new_node . data = new_data NEW_LINE new_node . next = ( head_ref ) NEW_LINE ( head_ref ) = new_node NEW_LINE return head_ref NEW_LINE DEDENT
def printList ( node ) : NEW_LINE INDENT while ( node != None ) : NEW_LINE INDENT print ( node . data , end = ' ▁ ' ) NEW_LINE node = node . next NEW_LINE DEDENT DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE
head = None NEW_LINE
for i in range ( 9 , 0 , - 1 ) : NEW_LINE INDENT head = push ( head , i ) NEW_LINE DEDENT print ( " Given ▁ linked ▁ list ▁ " ) NEW_LINE printList ( head ) NEW_LINE
k = 3 NEW_LINE d = 2 NEW_LINE head = rotateByBlocks ( head , k , d ) NEW_LINE print ( " Rotated by blocks Linked list   " ) NEW_LINE printList ( head ) NEW_LINE
def DeleteLast ( head ) : NEW_LINE INDENT current = head NEW_LINE temp = head NEW_LINE previous = None NEW_LINE DEDENT
if ( head == None ) : NEW_LINE INDENT print ( " List is empty " ) NEW_LINE return None NEW_LINE DEDENT
if ( current . next == current ) : NEW_LINE INDENT head = None NEW_LINE return None NEW_LINE DEDENT
while ( current . next != head ) : NEW_LINE INDENT previous = current NEW_LINE current = current . next NEW_LINE DEDENT previous . next = current . next NEW_LINE head = previous . next NEW_LINE return head NEW_LINE
def countSubarrays ( arr , n ) : NEW_LINE
count = 0 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT sum = 0 NEW_LINE for j in range ( i , n ) : NEW_LINE DEDENT
if ( ( j - i ) % 2 == 0 ) : NEW_LINE INDENT sum += arr [ j ] NEW_LINE DEDENT
else : NEW_LINE INDENT sum -= arr [ j ] NEW_LINE DEDENT
if ( sum == 0 ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT
print ( count ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
arr = [ 2 , 4 , 6 , 4 , 2 ] NEW_LINE
n = len ( arr ) NEW_LINE
countSubarrays ( arr , n ) NEW_LINE
def prints ( n ) : NEW_LINE INDENT if ( n < 0 ) : NEW_LINE INDENT return NEW_LINE DEDENT print ( str ( n ) , end = ' ▁ ' ) NEW_LINE DEDENT
prints ( n - 1 ) NEW_LINE
def printAlter ( arr , N ) : NEW_LINE
for currIndex in range ( 0 , N ) : NEW_LINE
if ( currIndex % 2 == 0 ) : NEW_LINE INDENT print ( arr [ currIndex ] , end = " ▁ " ) NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 1 , 2 , 3 , 4 , 5 ] NEW_LINE N = len ( arr ) NEW_LINE printAlter ( arr , N ) NEW_LINE DEDENT
' NEW_LINE def reverse ( arr , start , end ) : NEW_LINE
mid = ( end - start + 1 ) // 2 NEW_LINE
for i in range ( mid ) : NEW_LINE
temp = arr [ start + i ] NEW_LINE
arr [ start + i ] = arr [ end - i ] NEW_LINE
arr [ end - i ] = temp NEW_LINE return arr NEW_LINE
def shuffleArrayUtil ( arr , start , end ) : NEW_LINE INDENT i = 0 NEW_LINE DEDENT
l = end - start + 1 NEW_LINE
if ( l == 2 ) : NEW_LINE INDENT return NEW_LINE DEDENT
mid = start + l // 2 NEW_LINE
if ( l % 4 ) : NEW_LINE
mid -= 1 NEW_LINE
mid1 = start + ( mid - start ) // 2 NEW_LINE mid2 = mid + ( end + 1 - mid ) // 2 NEW_LINE
arr = reverse ( arr , mid1 , mid2 - 1 ) NEW_LINE
arr = reverse ( arr , mid1 , mid - 1 ) NEW_LINE
arr = reverse ( arr , mid , mid2 - 1 ) NEW_LINE
shuffleArrayUtil ( arr , start , mid - 1 ) NEW_LINE shuffleArrayUtil ( arr , mid , end ) NEW_LINE
def shuffleArray ( arr , N , start , end ) : NEW_LINE
shuffleArrayUtil ( arr , start , end ) NEW_LINE
for i in arr : NEW_LINE INDENT print ( i , end = " ▁ " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE
arr = [ 1 , 3 , 5 , 2 , 4 , 6 ] NEW_LINE
N = len ( arr ) NEW_LINE
shuffleArray ( arr , N , 0 , N - 1 ) NEW_LINE
def canMadeEqual ( A , B , n ) : NEW_LINE
INDENT A . sort ( ) NEW_LINE B . sort ( ) NEW_LINE DEDENT
INDENT for i in range ( n ) : NEW_LINE INDENT if ( A [ i ] != B [ i ] ) : NEW_LINE return False NEW_LINE DEDENT return True NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT A = [ 1 , 2 , 3 ] NEW_LINE B = [ 1 , 3 , 2 ] NEW_LINE n = len ( A ) NEW_LINE if ( canMadeEqual ( A , B , n ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT DEDENT
def merge ( arr , start , mid , end ) : NEW_LINE INDENT start2 = mid + 1 NEW_LINE DEDENT
if ( arr [ mid ] <= arr [ start2 ] ) : NEW_LINE INDENT return NEW_LINE DEDENT
while ( start <= mid and start2 <= end ) : NEW_LINE
if ( arr [ start ] <= arr [ start2 ] ) : NEW_LINE INDENT start += 1 NEW_LINE DEDENT else : NEW_LINE INDENT value = arr [ start2 ] NEW_LINE index = start2 NEW_LINE DEDENT
while ( index != start ) : NEW_LINE INDENT arr [ index ] = arr [ index - 1 ] NEW_LINE index -= 1 NEW_LINE DEDENT arr [ start ] = value NEW_LINE
start += 1 NEW_LINE mid += 1 NEW_LINE start2 += 1 NEW_LINE
def mergeSort ( arr , l , r ) : NEW_LINE INDENT if ( l < r ) : NEW_LINE DEDENT
m = l + ( r - l ) // 2 NEW_LINE
mergeSort ( arr , l , m ) NEW_LINE mergeSort ( arr , m + 1 , r ) NEW_LINE merge ( arr , l , m , r ) NEW_LINE
def printArray ( A , size ) : NEW_LINE INDENT for i in range ( size ) : NEW_LINE INDENT print ( A [ i ] , end = " ▁ " ) NEW_LINE DEDENT print ( ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 12 , 11 , 13 , 5 , 6 , 7 ] NEW_LINE arr_size = len ( arr ) NEW_LINE mergeSort ( arr , 0 , arr_size - 1 ) NEW_LINE printArray ( arr , arr_size ) NEW_LINE DEDENT
def dualPivotQuickSort ( arr , low , high ) : NEW_LINE INDENT if low < high : NEW_LINE DEDENT
lp , rp = partition ( arr , low , high ) NEW_LINE dualPivotQuickSort ( arr , low , lp - 1 ) NEW_LINE dualPivotQuickSort ( arr , lp + 1 , rp - 1 ) NEW_LINE dualPivotQuickSort ( arr , rp + 1 , high ) NEW_LINE def partition ( arr , low , high ) : NEW_LINE if arr [ low ] > arr [ high ] : NEW_LINE arr [ low ] , arr [ high ] = arr [ high ] , arr [ low ] NEW_LINE
j = k = low + 1 NEW_LINE g , p , q = high - 1 , arr [ low ] , arr [ high ] NEW_LINE while k <= g : NEW_LINE
if arr [ k ] < p : NEW_LINE INDENT arr [ k ] , arr [ j ] = arr [ j ] , arr [ k ] NEW_LINE j += 1 NEW_LINE DEDENT
elif arr [ k ] >= q : NEW_LINE INDENT while arr [ g ] > q and k < g : NEW_LINE INDENT g -= 1 NEW_LINE DEDENT arr [ k ] , arr [ g ] = arr [ g ] , arr [ k ] NEW_LINE g -= 1 NEW_LINE if arr [ k ] < p : NEW_LINE INDENT arr [ k ] , arr [ j ] = arr [ j ] , arr [ k ] NEW_LINE j += 1 NEW_LINE DEDENT DEDENT k += 1 NEW_LINE j -= 1 NEW_LINE g += 1 NEW_LINE
arr [ low ] , arr [ j ] = arr [ j ] , arr [ low ] NEW_LINE arr [ high ] , arr [ g ] = arr [ g ] , arr [ high ] NEW_LINE
return j , g NEW_LINE
arr = [ 24 , 8 , 42 , 75 , 29 , 77 , 38 , 57 ] NEW_LINE dualPivotQuickSort ( arr , 0 , 7 ) NEW_LINE print ( ' Sorted ▁ array : ▁ ' , end = ' ' ) NEW_LINE for i in arr : NEW_LINE INDENT print ( i , end = ' ▁ ' ) NEW_LINE DEDENT print ( ) NEW_LINE
def constGraphWithCon ( N , K ) : NEW_LINE
Max = ( ( N - 1 ) * ( N - 2 ) ) // 2 NEW_LINE
if ( K > Max ) : NEW_LINE INDENT print ( - 1 ) NEW_LINE return NEW_LINE DEDENT
ans = [ ] NEW_LINE
for i in range ( 1 , N ) : NEW_LINE INDENT for j in range ( i + 1 , N + 1 ) : NEW_LINE INDENT ans . append ( [ i , j ] ) NEW_LINE DEDENT DEDENT
for i in range ( 0 , ( N - 1 ) + Max - K ) : NEW_LINE INDENT print ( ans [ i ] [ 0 ] , ans [ i ] [ 1 ] , sep = " ▁ " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 5 NEW_LINE K = 3 NEW_LINE constGraphWithCon ( N , K ) NEW_LINE DEDENT
def findArray ( N , K ) : NEW_LINE
if ( N == 1 ) : NEW_LINE INDENT print ( K , end = " ▁ " ) NEW_LINE return NEW_LINE DEDENT if ( N == 2 ) : NEW_LINE INDENT print ( "0" , end = " ▁ " ) NEW_LINE print ( K , end = " ▁ " ) NEW_LINE return NEW_LINE DEDENT
P = N - 2 NEW_LINE Q = N - 1 NEW_LINE
VAL = 0 NEW_LINE
for i in range ( 1 , N - 2 ) : NEW_LINE INDENT print ( i , end = " ▁ " ) NEW_LINE DEDENT
VAL ^= i NEW_LINE if ( VAL == K ) : NEW_LINE print ( P , end = " ▁ " ) NEW_LINE print ( Q , end = " ▁ " ) NEW_LINE print ( P ^ Q , end = " ▁ " ) NEW_LINE else : NEW_LINE print ( "0" , end = " ▁ " ) NEW_LINE print ( P , end = " ▁ " ) NEW_LINE print ( P ^ K ^ VAL , end = " ▁ " ) NEW_LINE
N = 4 NEW_LINE X = 6 NEW_LINE
findArray ( N , X ) NEW_LINE
def countDigitSum ( N , K ) : NEW_LINE
l = pow ( 10 , N - 1 ) NEW_LINE r = pow ( 10 , N ) - 1 NEW_LINE count = 0 NEW_LINE for i in range ( l , r + 1 ) : NEW_LINE INDENT num = i NEW_LINE DEDENT
digits = [ 0 ] * N NEW_LINE for j in range ( N - 1 , - 1 , - 1 ) : NEW_LINE INDENT digits [ j ] = num % 10 NEW_LINE num //= 10 NEW_LINE DEDENT sum = 0 NEW_LINE flag = 0 NEW_LINE
for j in range ( 0 , K ) : NEW_LINE INDENT sum += digits [ j ] NEW_LINE DEDENT
for j in range ( 1 , N - K + 1 ) : NEW_LINE INDENT curr_sum = 0 NEW_LINE for m in range ( j , j + K ) : NEW_LINE INDENT curr_sum += digits [ m ] NEW_LINE DEDENT DEDENT
if ( sum != curr_sum ) : NEW_LINE INDENT flag = 1 NEW_LINE break NEW_LINE DEDENT
if ( flag == 0 ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT return count NEW_LINE
N = 2 NEW_LINE K = 1 NEW_LINE
print ( countDigitSum ( N , K ) ) NEW_LINE
def convert ( s ) : NEW_LINE
num = 0 NEW_LINE n = len ( s ) NEW_LINE
for i in s : NEW_LINE
num = num * 10 + ( ord ( i ) - 48 ) NEW_LINE
print ( num ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
s = "123" NEW_LINE
convert ( s ) NEW_LINE
def arithematicThree ( set_ , n ) : NEW_LINE
for j in range ( n ) : NEW_LINE
i , k = j - 1 , j + 1 NEW_LINE
while i > - 1 and k < n : NEW_LINE INDENT if set_ [ i ] + set_ [ k ] == 2 * set_ [ j ] : NEW_LINE INDENT return True NEW_LINE DEDENT elif set_ [ i ] + set_ [ k ] < 2 * set_ [ j ] : NEW_LINE INDENT i -= 1 NEW_LINE DEDENT else : NEW_LINE INDENT k += 1 NEW_LINE DEDENT DEDENT return False NEW_LINE
def maxSumIS ( arr , n ) : NEW_LINE INDENT max = 0 NEW_LINE msis = [ 0 for x in range ( n ) ] NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT msis [ i ] = arr [ i ] NEW_LINE DEDENT
for i in range ( 1 , n ) : NEW_LINE INDENT for j in range ( i ) : NEW_LINE INDENT if ( arr [ i ] > arr [ j ] and msis [ i ] < msis [ j ] + arr [ i ] ) : NEW_LINE INDENT msis [ i ] = msis [ j ] + arr [ i ] NEW_LINE DEDENT DEDENT DEDENT
for i in range ( n ) : NEW_LINE INDENT if max < msis [ i ] : NEW_LINE INDENT max = msis [ i ] NEW_LINE DEDENT DEDENT return max NEW_LINE
arr = [ 1 , 101 , 2 , 3 , 100 , 4 , 5 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Sum ▁ of ▁ maximum ▁ sum ▁ increasing ▁ " + " subsequence ▁ is ▁ " + str ( maxSumIS ( arr , n ) ) ) NEW_LINE
def fib ( n ) : NEW_LINE INDENT if n <= 1 : NEW_LINE INDENT return n NEW_LINE DEDENT return fib ( n - 1 ) + fib ( n - 2 ) NEW_LINE DEDENT
def internalSearch ( ii , needle , row , col , hay , row_max , col_max ) : NEW_LINE INDENT found = 0 NEW_LINE if ( row >= 0 and row <= row_max and col >= 0 and col <= col_max and needle [ ii ] == hay [ row ] [ col ] ) : NEW_LINE INDENT match = needle [ ii ] NEW_LINE ii += 1 NEW_LINE hay [ row ] [ col ] = 0 NEW_LINE if ( ii == len ( needle ) ) : NEW_LINE INDENT found = 1 NEW_LINE DEDENT else : NEW_LINE DEDENT DEDENT
found += internalSearch ( ii , needle , row , col + 1 , hay , row_max , col_max ) NEW_LINE found += internalSearch ( ii , needle , row , col - 1 , hay , row_max , col_max ) NEW_LINE found += internalSearch ( ii , needle , row + 1 , col , hay , row_max , col_max ) NEW_LINE found += internalSearch ( ii , needle , row - 1 , col , hay , row_max , col_max ) NEW_LINE hay [ row ] [ col ] = match NEW_LINE return found NEW_LINE
def searchString ( needle , row , col , strr , row_count , col_count ) : NEW_LINE INDENT found = 0 NEW_LINE for r in range ( row_count ) : NEW_LINE INDENT for c in range ( col_count ) : NEW_LINE INDENT found += internalSearch ( 0 , needle , r , c , strr , row_count - 1 , col_count - 1 ) NEW_LINE DEDENT DEDENT return found NEW_LINE DEDENT
needle = " MAGIC " NEW_LINE inputt = [ " BBABBM " , " CBMBBA " , " IBABBG " , " GOZBBI " , " ABBBBC " , " MCIGAM " ] NEW_LINE strr = [ 0 ] * len ( inputt ) NEW_LINE for i in range ( len ( inputt ) ) : NEW_LINE INDENT strr [ i ] = list ( inputt [ i ] ) NEW_LINE DEDENT print ( " count : ▁ " , searchString ( needle , 0 , 0 , strr , len ( strr ) , len ( strr [ 0 ] ) ) ) NEW_LINE
def isBalanced ( exp ) : NEW_LINE
flag = True NEW_LINE count = 0 NEW_LINE
for i in range ( len ( exp ) ) : NEW_LINE INDENT if ( exp [ i ] == ' ( ' ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT else : NEW_LINE DEDENT
count -= 1 NEW_LINE if ( count < 0 ) : NEW_LINE
flag = False NEW_LINE break NEW_LINE
if ( count != 0 ) : NEW_LINE INDENT flag = False NEW_LINE DEDENT return flag NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT exp1 = " ( ( ( ) ) ) ( ) ( ) " NEW_LINE if ( isBalanced ( exp1 ) ) : NEW_LINE INDENT print ( " Balanced " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Not ▁ Balanced " ) NEW_LINE DEDENT exp2 = " ( ) ) ( ( ( ) ) " NEW_LINE if ( isBalanced ( exp2 ) ) : NEW_LINE INDENT print ( " Balanced " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Not ▁ Balanced " ) NEW_LINE DEDENT DEDENT
def reverse ( string , start , end ) : NEW_LINE
temp = ' ' NEW_LINE while start <= end : NEW_LINE
temp = string [ start ] NEW_LINE string [ start ] = string [ end ] NEW_LINE string [ end ] = temp NEW_LINE start += 1 NEW_LINE end -= 1 NEW_LINE
def reverseletter ( string , start , end ) : NEW_LINE INDENT wstart , wend = start , start NEW_LINE while wend < end : NEW_LINE INDENT if string [ wend ] == " ▁ " : NEW_LINE INDENT wend += 1 NEW_LINE continue NEW_LINE DEDENT DEDENT DEDENT
while wend <= end and string [ wend ] != " ▁ " : NEW_LINE INDENT wend += 1 NEW_LINE DEDENT wend -= 1 NEW_LINE
reverse ( string , wstart , wend ) NEW_LINE wend += 1 NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT string = " Ashish ▁ Yadav ▁ Abhishek ▁ Rajput ▁ Sunil ▁ Pundir " NEW_LINE string = list ( string ) NEW_LINE reverseletter ( string , 0 , len ( string ) - 1 ) NEW_LINE print ( ' ' . join ( string ) ) NEW_LINE DEDENT
from collections import defaultdict NEW_LINE def have_same_frequency ( freq : defaultdict , k : int ) : NEW_LINE INDENT return all ( [ freq [ i ] == k or freq [ i ] == 0 for i in freq ] ) NEW_LINE DEDENT def count_substrings ( s : str , k : int ) -> int : NEW_LINE INDENT count = 0 NEW_LINE distinct = len ( set ( [ i for i in s ] ) ) NEW_LINE for length in range ( 1 , distinct + 1 ) : NEW_LINE INDENT window_length = length * k NEW_LINE freq = defaultdict ( int ) NEW_LINE window_start = 0 NEW_LINE window_end = window_start + window_length - 1 NEW_LINE for i in range ( window_start , min ( window_end + 1 , len ( s ) ) ) : NEW_LINE INDENT freq [ s [ i ] ] += 1 NEW_LINE DEDENT while window_end < len ( s ) : NEW_LINE INDENT if have_same_frequency ( freq , k ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT freq [ s [ window_start ] ] -= 1 NEW_LINE window_start += 1 NEW_LINE window_end += 1 NEW_LINE if window_end < len ( s ) : NEW_LINE INDENT freq [ s [ window_end ] ] += 1 NEW_LINE DEDENT DEDENT DEDENT return count NEW_LINE DEDENT if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT s = " aabbcc " NEW_LINE k = 2 NEW_LINE print ( count_substrings ( s , k ) ) NEW_LINE s = " aabbc " NEW_LINE k = 2 NEW_LINE print ( count_substrings ( s , k ) ) NEW_LINE DEDENT
x = 32 ; NEW_LINE
def toggleCase ( a ) : NEW_LINE INDENT for i in range ( len ( a ) ) : NEW_LINE DEDENT
a = a [ : i ] + chr ( ord ( a [ i ] ) ^ 32 ) + a [ i + 1 : ] ; NEW_LINE return a ; NEW_LINE
str = " CheRrY " ; NEW_LINE print ( " Toggle ▁ case : ▁ " , end = " " ) ; NEW_LINE str = toggleCase ( str ) ; NEW_LINE print ( str ) ; NEW_LINE print ( " Original ▁ string : ▁ " , end = " " ) ; NEW_LINE str = toggleCase ( str ) ; NEW_LINE print ( str ) ; NEW_LINE
NO_OF_CHARS = 256 NEW_LINE
def areAnagram ( str1 , str2 ) : NEW_LINE
count1 = [ 0 ] * NO_OF_CHARS NEW_LINE count2 = [ 0 ] * NO_OF_CHARS NEW_LINE
for i in str1 : NEW_LINE INDENT count1 [ ord ( i ) ] += 1 NEW_LINE DEDENT for i in str2 : NEW_LINE INDENT count2 [ ord ( i ) ] += 1 NEW_LINE DEDENT
if len ( str1 ) != len ( str2 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
for i in xrange ( NO_OF_CHARS ) : NEW_LINE INDENT if count1 [ i ] != count2 [ i ] : NEW_LINE INDENT return 0 NEW_LINE DEDENT DEDENT return 1 NEW_LINE
str1 = " geeksforgeeks " NEW_LINE str2 = " forgeeksgeeks " NEW_LINE
if areAnagram ( str1 , str2 ) : NEW_LINE INDENT print " The ▁ two ▁ strings ▁ are ▁ anagram ▁ of ▁ each ▁ other " NEW_LINE DEDENT else : NEW_LINE INDENT print " The ▁ two ▁ strings ▁ are ▁ not ▁ anagram ▁ of ▁ each ▁ other " NEW_LINE DEDENT
def heptacontagonNum ( n ) : NEW_LINE INDENT return ( 68 * n * n - 66 * n ) // 2 ; NEW_LINE DEDENT
N = 3 ; NEW_LINE print ( "3rd ▁ heptacontagon ▁ Number ▁ is ▁ = " , heptacontagonNum ( N ) ) ; NEW_LINE
def isEqualFactors ( N ) : NEW_LINE INDENT if ( ( N % 2 == 0 ) and ( N % 4 != 0 ) ) : NEW_LINE INDENT print ( " YES " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NO " ) NEW_LINE DEDENT DEDENT
N = 10 NEW_LINE isEqualFactors ( N ) NEW_LINE N = 125 ; NEW_LINE isEqualFactors ( N ) NEW_LINE
import math NEW_LINE
def checkDivisibility ( n , digit ) : NEW_LINE
return ( ( digit != 0 ) and ( ( n % digit ) == 0 ) ) NEW_LINE
def isAllDigitsDivide ( n ) : NEW_LINE INDENT temp = n NEW_LINE while ( temp >= 1 ) : NEW_LINE DEDENT
digit = int ( temp % 10 ) NEW_LINE if ( checkDivisibility ( n , digit ) == False ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT temp = temp / 10 NEW_LINE return 1 NEW_LINE
def isAllDigitsDistinct ( n ) : NEW_LINE
arr = [ 0 ] * 10 NEW_LINE
while ( n >= 1 ) : NEW_LINE
digit = int ( n % 10 ) NEW_LINE
if ( arr [ digit ] ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
arr [ digit ] = 1 NEW_LINE
n = int ( n / 10 ) NEW_LINE return 1 NEW_LINE
def isLynchBell ( n ) : NEW_LINE INDENT return ( isAllDigitsDivide ( n ) and isAllDigitsDistinct ( n ) ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE
N = 12 NEW_LINE
if isLynchBell ( N ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
def maximumAND ( L , R ) : NEW_LINE INDENT return R NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT l = 3 NEW_LINE r = 7 NEW_LINE print ( maximumAND ( l , r ) ) NEW_LINE DEDENT
def findAverageOfCube ( n ) : NEW_LINE
sum = 0 NEW_LINE
for i in range ( 1 , n + 1 ) : NEW_LINE INDENT sum += i * i * i NEW_LINE DEDENT
return round ( sum / n , 6 ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
n = 3 NEW_LINE
print ( findAverageOfCube ( n ) ) NEW_LINE
from math import log NEW_LINE
def isPower ( n , k ) : NEW_LINE
res1 = int ( log ( n ) / log ( k ) ) NEW_LINE res2 = log ( n ) / log ( k ) NEW_LINE
return ( res1 == res2 ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 8 NEW_LINE k = 2 NEW_LINE if ( isPower ( n , k ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT DEDENT
def y ( x ) : NEW_LINE INDENT return ( 1 / ( 1 + x ) ) NEW_LINE DEDENT
def BooleRule ( a , b ) : NEW_LINE
n = 4 NEW_LINE
h = ( ( b - a ) / n ) NEW_LINE sum = 0 NEW_LINE
bl = ( 7 * y ( a ) + 32 * y ( a + h ) + 12 * y ( a + 2 * h ) + 32 * y ( a + 3 * h ) + 7 * y ( a + 4 * h ) ) * 2 * h / 45 NEW_LINE sum = sum + bl NEW_LINE return sum NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT lowlimit = 0 NEW_LINE upplimit = 4 NEW_LINE print ( " f ( x ) ▁ = " , round ( BooleRule ( 0 , 4 ) , 4 ) ) NEW_LINE DEDENT
def y ( x ) : NEW_LINE INDENT num = 1 ; NEW_LINE denom = float ( 1.0 + x * x ) ; NEW_LINE return num / denom ; NEW_LINE DEDENT
def WeedleRule ( a , b ) : NEW_LINE
h = ( b - a ) / 6 ; NEW_LINE
sum = 0 ; NEW_LINE
sum = sum + ( ( ( 3 * h ) / 10 ) * ( y ( a ) + y ( a + 2 * h ) + 5 * y ( a + h ) + 6 * y ( a + 3 * h ) + y ( a + 4 * h ) + 5 * y ( a + 5 * h ) + y ( a + 6 * h ) ) ) ; NEW_LINE
return sum ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
a = 0 ; NEW_LINE b = 6 ; NEW_LINE
num = WeedleRule ( a , b ) ; NEW_LINE print ( " f ( x ) ▁ = ▁ { 0 : . 6f } " . format ( num ) ) ; NEW_LINE
def dydx ( x , y ) : NEW_LINE INDENT return ( x + y - 2 ) ; NEW_LINE DEDENT
def rungeKutta ( x0 , y0 , x , h ) : NEW_LINE
n = round ( ( x - x0 ) / h ) ; NEW_LINE
y = y0 ; NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE
k1 = h * dydx ( x0 , y ) ; NEW_LINE k2 = h * dydx ( x0 + 0.5 * h , y + 0.5 * k1 ) ; NEW_LINE
y = y + ( 1.0 / 6.0 ) * ( k1 + 2 * k2 ) ; NEW_LINE
x0 = x0 + h ; NEW_LINE return y ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT x0 = 0 ; y = 1 ; NEW_LINE x = 2 ; h = 0.2 ; NEW_LINE print ( " y ( x ) ▁ = " , rungeKutta ( x0 , y , x , h ) ) ; NEW_LINE DEDENT
def per ( a , b ) : NEW_LINE INDENT return ( a + b ) NEW_LINE DEDENT
def area ( s ) : NEW_LINE INDENT return ( s / 2 ) NEW_LINE DEDENT
a = 7 NEW_LINE b = 8 NEW_LINE s = 10 NEW_LINE print ( per ( a , b ) ) NEW_LINE print ( area ( s ) ) NEW_LINE
PI = 3.14159265 NEW_LINE
def area_leaf ( a ) : NEW_LINE INDENT return ( a * a * ( PI / 2 - 1 ) ) NEW_LINE DEDENT
a = 7 NEW_LINE print ( area_leaf ( a ) ) NEW_LINE
PI = 3.14159265 NEW_LINE
def length_rope ( r ) : NEW_LINE INDENT return ( ( 2 * PI * r ) + 6 * r ) NEW_LINE DEDENT
r = 7 NEW_LINE print ( length_rope ( r ) ) NEW_LINE
PI = 3.14159265 NEW_LINE
def area_inscribed ( P , B , H ) : NEW_LINE INDENT return ( ( P + B - H ) * ( P + B - H ) * ( PI / 4 ) ) NEW_LINE DEDENT
P = 3 NEW_LINE B = 4 NEW_LINE H = 5 NEW_LINE print ( area_inscribed ( P , B , H ) ) NEW_LINE
PI = 3.14159265 NEW_LINE
def area_cicumscribed ( c ) : NEW_LINE INDENT return ( c * c * ( PI / 4 ) ) NEW_LINE DEDENT
c = 8.0 NEW_LINE print ( area_cicumscribed ( c ) ) NEW_LINE
import math NEW_LINE PI = 3.14159265 NEW_LINE
def area_inscribed ( a ) : NEW_LINE INDENT return ( a * a * ( PI / 12 ) ) NEW_LINE DEDENT
def perm_inscribed ( a ) : NEW_LINE INDENT return ( PI * ( a / math . sqrt ( 3 ) ) ) NEW_LINE DEDENT
a = 6.0 NEW_LINE print ( " Area ▁ of ▁ inscribed ▁ circle ▁ is ▁ : % ▁ f " % area_inscribed ( a ) ) NEW_LINE print ( " Perimeter of inscribed circle is : % f " % perm_inscribed ( a ) ) NEW_LINE
def area ( r ) : NEW_LINE
return ( 0.5 ) * ( 3.14 ) * ( r * r ) NEW_LINE
def perimeter ( r ) : NEW_LINE
return ( 3.14 ) * ( r ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
r = 10 NEW_LINE
print ( " The ▁ Area ▁ of ▁ Semicircle : ▁ " , area ( r ) ) NEW_LINE
print ( " The ▁ Perimeter ▁ of ▁ Semicircle : ▁ " , perimeter ( r ) ) NEW_LINE
def equation_plane ( x1 , y1 , z1 , x2 , y2 , z2 , x3 , y3 , z3 ) : NEW_LINE INDENT a1 = x2 - x1 NEW_LINE b1 = y2 - y1 NEW_LINE c1 = z2 - z1 NEW_LINE a2 = x3 - x1 NEW_LINE b2 = y3 - y1 NEW_LINE c2 = z3 - z1 NEW_LINE a = b1 * c2 - b2 * c1 NEW_LINE b = a2 * c1 - a1 * c2 NEW_LINE c = a1 * b2 - b1 * a2 NEW_LINE d = ( - a * x1 - b * y1 - c * z1 ) NEW_LINE print " equation ▁ of ▁ plane ▁ is ▁ " , NEW_LINE print a , " x ▁ + " , NEW_LINE print b , " y ▁ + " , NEW_LINE print c , " z ▁ + " , NEW_LINE print d , " = ▁ 0 . " NEW_LINE DEDENT
x1 = - 1 NEW_LINE y1 = 2 NEW_LINE z1 = 1 NEW_LINE x2 = 0 NEW_LINE y2 = - 3 NEW_LINE z2 = 2 NEW_LINE x3 = 1 NEW_LINE y3 = 1 NEW_LINE z3 = - 4 NEW_LINE equation_plane ( x1 , y1 , z1 , x2 , y2 , z2 , x3 , y3 , z3 ) NEW_LINE
import math NEW_LINE
def shortest_distance ( x1 , y1 , a , b , c ) : NEW_LINE INDENT d = abs ( ( a * x1 + b * y1 + c ) ) / ( math . sqrt ( a * a + b * b ) ) NEW_LINE print ( " Perpendicular ▁ distance ▁ is " ) , d NEW_LINE DEDENT
x1 = 5 NEW_LINE y1 = 6 NEW_LINE a = - 2 NEW_LINE b = 3 NEW_LINE c = 4 NEW_LINE shortest_distance ( x1 , y1 , a , b , c ) NEW_LINE
def octant ( x , y , z ) : NEW_LINE INDENT if x >= 0 and y >= 0 and z >= 0 : NEW_LINE INDENT print " Point ▁ lies ▁ in ▁ 1st ▁ octant " NEW_LINE DEDENT elif x < 0 and y >= 0 and z >= 0 : NEW_LINE INDENT print " Point ▁ lies ▁ in ▁ 2nd ▁ octant " NEW_LINE DEDENT elif x < 0 and y < 0 and z >= 0 : NEW_LINE INDENT print " Point ▁ lies ▁ in ▁ 3rd ▁ octant " NEW_LINE DEDENT elif x >= 0 and y < 0 and z >= 0 : NEW_LINE INDENT print " Point ▁ lies ▁ in ▁ 4th ▁ octant " NEW_LINE DEDENT elif x >= 0 and y >= 0 and z < 0 : NEW_LINE INDENT print " Point ▁ lies ▁ in ▁ 5th ▁ octant " NEW_LINE DEDENT elif x < 0 and y >= 0 and z < 0 : NEW_LINE INDENT print " Point ▁ lies ▁ in ▁ 6th ▁ octant " NEW_LINE DEDENT elif x < 0 and y < 0 and z < 0 : NEW_LINE INDENT print " Point ▁ lies ▁ in ▁ 7th ▁ octant " NEW_LINE DEDENT elif x >= 0 and y < 0 and z < 0 : NEW_LINE INDENT print " Point ▁ lies ▁ in ▁ 8th ▁ octant " NEW_LINE DEDENT DEDENT
x , y , z = 2 , 3 , 4 NEW_LINE octant ( x , y , z ) NEW_LINE x , y , z = - 4 , 2 , - 8 NEW_LINE octant ( x , y , z ) NEW_LINE x , y , z = - 6 , - 2 , 8 NEW_LINE octant ( x , y , z ) NEW_LINE
import math NEW_LINE def maxArea ( a , b , c , d ) : NEW_LINE
semiperimeter = ( a + b + c + d ) / 2 NEW_LINE
return math . sqrt ( ( semiperimeter - a ) * ( semiperimeter - b ) * ( semiperimeter - c ) * ( semiperimeter - d ) ) NEW_LINE
a = 1 NEW_LINE b = 2 NEW_LINE c = 1 NEW_LINE d = 2 NEW_LINE print ( " % .2f " % maxArea ( a , b , c , d ) ) NEW_LINE
def addAP ( A , Q , operations ) : NEW_LINE
for L , R , a , d in operations : NEW_LINE INDENT curr = a NEW_LINE DEDENT
for i in range ( L - 1 , R ) : NEW_LINE
A [ i ] += curr NEW_LINE
curr += d NEW_LINE
for i in A : NEW_LINE INDENT print ( i , end = ' ▁ ' ) NEW_LINE DEDENT
A = [ 5 , 4 , 2 , 8 ] NEW_LINE Q = 2 NEW_LINE Query = [ ( 1 , 2 , 1 , 3 ) , ( 1 , 4 , 4 , 1 ) ] NEW_LINE
addAP ( A , Q , Query ) NEW_LINE
from math import log NEW_LINE def log_a_to_base_b ( a , b ) : NEW_LINE INDENT return log ( a ) // log ( b ) ; NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = 3 ; NEW_LINE b = 2 ; NEW_LINE print ( log_a_to_base_b ( a , b ) ) ; NEW_LINE a = 256 ; NEW_LINE b = 4 ; NEW_LINE print ( log_a_to_base_b ( a , b ) ) ; NEW_LINE DEDENT
def log_a_to_base_b ( a , b ) : NEW_LINE INDENT rslt = ( 1 + log_a_to_base_b ( a // b , b ) ) if ( a > ( b - 1 ) ) else 0 ; NEW_LINE return rslt ; NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = 3 ; NEW_LINE b = 2 ; NEW_LINE print ( log_a_to_base_b ( a , b ) ) ; NEW_LINE a = 256 ; NEW_LINE b = 4 ; NEW_LINE print ( log_a_to_base_b ( a , b ) ) ; NEW_LINE DEDENT
def maximum ( x , y ) : NEW_LINE INDENT return ( ( x + y + abs ( x - y ) ) // 2 ) NEW_LINE DEDENT
def minimum ( x , y ) : NEW_LINE INDENT return ( ( x + y - abs ( x - y ) ) // 2 ) NEW_LINE DEDENT
x = 99 NEW_LINE y = 18 NEW_LINE
print ( " Maximum : " , maximum ( x , y ) ) NEW_LINE
print ( " Minimum : " , minimum ( x , y ) ) NEW_LINE
p = 1.0 NEW_LINE f = 1.0 NEW_LINE def e ( x , n ) : NEW_LINE INDENT global p , f NEW_LINE DEDENT
if ( n == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT
r = e ( x , n - 1 ) NEW_LINE
p = p * x NEW_LINE
f = f * n NEW_LINE return ( r + p / f ) NEW_LINE
x = 4 NEW_LINE n = 15 NEW_LINE print ( e ( x , n ) ) NEW_LINE
def midptellipse ( rx , ry , xc , yc ) : NEW_LINE INDENT x = 0 ; NEW_LINE y = ry ; NEW_LINE DEDENT
d1 = ( ( ry * ry ) - ( rx * rx * ry ) + ( 0.25 * rx * rx ) ) ; NEW_LINE dx = 2 * ry * ry * x ; NEW_LINE dy = 2 * rx * rx * y ; NEW_LINE
while ( dx < dy ) : NEW_LINE
print ( " ( " , x + xc , " , " , y + yc , " ) " ) ; NEW_LINE print ( " ( " , - x + xc , " , " , y + yc , " ) " ) ; NEW_LINE print ( " ( " , x + xc , " , " , - y + yc , " ) " ) ; NEW_LINE print ( " ( " , - x + xc , " , " , - y + yc , " ) " ) ; NEW_LINE
if ( d1 < 0 ) : NEW_LINE INDENT x += 1 ; NEW_LINE dx = dx + ( 2 * ry * ry ) ; NEW_LINE d1 = d1 + dx + ( ry * ry ) ; NEW_LINE DEDENT else : NEW_LINE INDENT x += 1 ; NEW_LINE y -= 1 ; NEW_LINE dx = dx + ( 2 * ry * ry ) ; NEW_LINE dy = dy - ( 2 * rx * rx ) ; NEW_LINE d1 = d1 + dx - dy + ( ry * ry ) ; NEW_LINE DEDENT
d2 = ( ( ( ry * ry ) * ( ( x + 0.5 ) * ( x + 0.5 ) ) ) + ( ( rx * rx ) * ( ( y - 1 ) * ( y - 1 ) ) ) - ( rx * rx * ry * ry ) ) ; NEW_LINE
while ( y >= 0 ) : NEW_LINE
print ( " ( " , x + xc , " , " , y + yc , " ) " ) ; NEW_LINE print ( " ( " , - x + xc , " , " , y + yc , " ) " ) ; NEW_LINE print ( " ( " , x + xc , " , " , - y + yc , " ) " ) ; NEW_LINE print ( " ( " , - x + xc , " , " , - y + yc , " ) " ) ; NEW_LINE
if ( d2 > 0 ) : NEW_LINE INDENT y -= 1 ; NEW_LINE dy = dy - ( 2 * rx * rx ) ; NEW_LINE d2 = d2 + ( rx * rx ) - dy ; NEW_LINE DEDENT else : NEW_LINE INDENT y -= 1 ; NEW_LINE x += 1 ; NEW_LINE dx = dx + ( 2 * ry * ry ) ; NEW_LINE dy = dy - ( 2 * rx * rx ) ; NEW_LINE d2 = d2 + dx - dy + ( rx * rx ) ; NEW_LINE DEDENT
midptellipse ( 10 , 15 , 50 , 50 ) ; NEW_LINE
def HexToBin ( hexdec ) : NEW_LINE INDENT for i in hexdec : NEW_LINE INDENT if i == '0' : NEW_LINE INDENT print ( '0000' , end = ' ' ) NEW_LINE DEDENT elif i == '1' : NEW_LINE INDENT print ( '0001' , end = ' ' ) NEW_LINE DEDENT elif i == '2' : NEW_LINE INDENT print ( '0010' , end = ' ' ) NEW_LINE DEDENT elif i == '3' : NEW_LINE INDENT print ( '0011' , end = ' ' ) NEW_LINE DEDENT elif i == '4' : NEW_LINE INDENT print ( '0100' , end = ' ' ) NEW_LINE DEDENT elif i == '5' : NEW_LINE INDENT print ( '0101' , end = ' ' ) NEW_LINE DEDENT elif i == '6' : NEW_LINE INDENT print ( '0110' , end = ' ' ) NEW_LINE DEDENT elif i == '7' : NEW_LINE INDENT print ( '0111' , end = ' ' ) NEW_LINE DEDENT elif i == '8' : NEW_LINE INDENT print ( '1000' , end = ' ' ) NEW_LINE DEDENT elif i == '9' : NEW_LINE INDENT print ( '1001' , end = ' ' ) NEW_LINE DEDENT elif i == ' A ' or i == ' a ' : NEW_LINE INDENT print ( '1010' , end = ' ' ) NEW_LINE DEDENT elif i == ' B ' or i == ' b ' : NEW_LINE INDENT print ( '1011' , end = ' ' ) NEW_LINE DEDENT elif i == ' C ' or i == ' c ' : NEW_LINE INDENT print ( '1100' , end = ' ' ) NEW_LINE DEDENT elif i == ' D ' or i == ' d ' : NEW_LINE INDENT print ( '1101' , end = ' ' ) NEW_LINE DEDENT elif i == ' E ' or i == ' e ' : NEW_LINE INDENT print ( '1110' , end = ' ' ) NEW_LINE DEDENT elif i == ' F ' or i == ' f ' : NEW_LINE INDENT print ( '1111' , end = ' ' ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Invalid hexadecimal digit   " + str ( hexdec [ i ] ) , end = ' ' ) NEW_LINE DEDENT DEDENT DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE
hexdec = "1AC5" ; NEW_LINE
print ( " Equivalent ▁ Binary ▁ value ▁ is ▁ : ▁ " , end = ' ' ) NEW_LINE HexToBin ( hexdec ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT matrix = [ [ 0 for i in range ( 5 ) ] for j in range ( 5 ) ] ; NEW_LINE row_index , column_index , x , size = 0 , 0 , 0 , 5 ; NEW_LINE DEDENT
for row_index in range ( size ) : NEW_LINE INDENT for column_index in range ( size ) : NEW_LINE INDENT x += 1 ; NEW_LINE matrix [ row_index ] [ column_index ] += x ; NEW_LINE DEDENT DEDENT
print ( " The ▁ matrix ▁ is " ) ; NEW_LINE for row_index in range ( size ) : NEW_LINE INDENT for column_index in range ( size ) : NEW_LINE INDENT print ( matrix [ row_index ] [ column_index ] , end = " TABSYMBOL " ) ; NEW_LINE DEDENT print ( " " ) ; NEW_LINE DEDENT
print ( " Elements above Secondary diagonal are : " ) ; NEW_LINE for row_index in range ( size ) : NEW_LINE INDENT for column_index in range ( size ) : NEW_LINE DEDENT
if ( ( row_index + column_index ) < size - 1 ) : NEW_LINE INDENT print ( matrix [ row_index ] [ column_index ] , end = " ▁ " ) ; NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT matrix = [ [ 0 for i in range ( 5 ) ] for j in range ( 5 ) ] NEW_LINE row_index , column_index , x , size = 0 , 0 , 0 , 5 ; NEW_LINE DEDENT
for row_index in range ( size ) : NEW_LINE INDENT for column_index in range ( size ) : NEW_LINE INDENT x += 1 ; NEW_LINE matrix [ row_index ] [ column_index ] = x ; NEW_LINE DEDENT DEDENT
print ( " The ▁ matrix ▁ is " ) ; NEW_LINE for row_index in range ( size ) : NEW_LINE INDENT for column_index in range ( size ) : NEW_LINE INDENT print ( matrix [ row_index ] [ column_index ] , end = " TABSYMBOL " ) ; NEW_LINE DEDENT print ( " " ) ; NEW_LINE DEDENT
print ( " Corner Elements are : " ) ; NEW_LINE for row_index in range ( size ) : NEW_LINE INDENT for column_index in range ( size ) : NEW_LINE DEDENT
if ( ( row_index == 0 or row_index == size - 1 ) and ( column_index == 0 or column_index == size - 1 ) ) : NEW_LINE INDENT print ( matrix [ row_index ] [ column_index ] , end = " TABSYMBOL " ) ; NEW_LINE DEDENT
import math NEW_LINE
def distance ( x1 , y1 , z1 , x2 , y2 , z2 ) : NEW_LINE INDENT d = math . sqrt ( math . pow ( x2 - x1 , 2 ) + math . pow ( y2 - y1 , 2 ) + math . pow ( z2 - z1 , 2 ) * 1.0 ) NEW_LINE print ( " Distance ▁ is ▁ " ) NEW_LINE print ( d ) NEW_LINE DEDENT
x1 = 2 NEW_LINE y1 = - 5 NEW_LINE z1 = 7 NEW_LINE x2 = 3 NEW_LINE y2 = 4 NEW_LINE z2 = 5 NEW_LINE
distance ( x1 , y1 , z1 , x2 , y2 , z2 ) NEW_LINE
def No_Of_Pairs ( N ) : NEW_LINE INDENT i = 1 ; NEW_LINE DEDENT
while ( ( i * i * i ) + ( 2 * i * i ) + i <= N ) : NEW_LINE INDENT i += 1 ; NEW_LINE DEDENT return ( i - 1 ) ; NEW_LINE
def print_pairs ( pairs ) : NEW_LINE INDENT i = 1 ; NEW_LINE mul = 0 ; NEW_LINE for i in range ( 1 , pairs + 1 ) : NEW_LINE INDENT mul = i * ( i + 1 ) ; NEW_LINE print ( " Pair ▁ no . " , i , " ▁ - - > ▁ ( " , ( mul * i ) , " , ▁ " , mul * ( i + 1 ) , " ) " ) ; NEW_LINE DEDENT DEDENT
N = 500 ; NEW_LINE i = 1 ; NEW_LINE pairs = No_Of_Pairs ( N ) ; NEW_LINE print ( " No . ▁ of ▁ pairs ▁ = ▁ " , pairs ) ; NEW_LINE print_pairs ( pairs ) ; NEW_LINE
def findArea ( d ) : NEW_LINE INDENT return ( d * d ) / 2 NEW_LINE DEDENT
d = 10 NEW_LINE print ( " % .2f " % findArea ( d ) ) NEW_LINE
def AvgofSquareN ( n ) : NEW_LINE INDENT sum = 0 NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE INDENT sum += ( i * i ) NEW_LINE DEDENT return sum / n NEW_LINE DEDENT
n = 2 NEW_LINE print ( AvgofSquareN ( n ) ) NEW_LINE
import math NEW_LINE
def Series ( x , n ) : NEW_LINE INDENT sum = 1 NEW_LINE term = 1 NEW_LINE y = 2 NEW_LINE DEDENT
for i in range ( 1 , n ) : NEW_LINE INDENT fct = 1 NEW_LINE for j in range ( 1 , y + 1 ) : NEW_LINE INDENT fct = fct * j NEW_LINE DEDENT term = term * ( - 1 ) NEW_LINE m = term * math . pow ( x , y ) / fct NEW_LINE sum = sum + m NEW_LINE y += 2 NEW_LINE DEDENT return sum NEW_LINE
x = 9 NEW_LINE n = 10 NEW_LINE print ( ' % .4f ' % Series ( x , n ) ) NEW_LINE
import math NEW_LINE
def maxPrimeFactors ( n ) : NEW_LINE
maxPrime = - 1 NEW_LINE
while n % 2 == 0 : NEW_LINE INDENT maxPrime = 2 NEW_LINE DEDENT
while n % 3 == 0 : NEW_LINE INDENT maxPrime = 3 NEW_LINE n = n / 3 NEW_LINE DEDENT
for i in range ( 5 , int ( math . sqrt ( n ) ) + 1 , 6 ) : NEW_LINE INDENT while n % i == 0 : NEW_LINE INDENT maxPrime = i NEW_LINE n = n / i NEW_LINE DEDENT while n % ( i + 2 ) == 0 : NEW_LINE INDENT maxPrime = i + 2 NEW_LINE n = n / ( i + 2 ) NEW_LINE DEDENT DEDENT
if n > 4 : NEW_LINE INDENT maxPrime = n NEW_LINE DEDENT return int ( maxPrime ) NEW_LINE
n = 15 NEW_LINE print ( maxPrimeFactors ( n ) ) NEW_LINE n = 25698751364526 NEW_LINE print ( maxPrimeFactors ( n ) ) NEW_LINE
def sum ( x , n ) : NEW_LINE INDENT total = 1.0 NEW_LINE multi = x NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE INDENT total = total + multi / i NEW_LINE multi = multi * x NEW_LINE DEDENT return total NEW_LINE DEDENT
x = 2 NEW_LINE n = 5 NEW_LINE print ( round ( sum ( x , n ) , 2 ) ) NEW_LINE
def chiliagonNum ( n ) : NEW_LINE INDENT return ( 998 * n * n - 996 * n ) // 2 ; NEW_LINE DEDENT
n = 3 ; NEW_LINE print ( "3rd ▁ chiliagon ▁ Number ▁ is ▁ = ▁ " , chiliagonNum ( n ) ) ; NEW_LINE
def pentacontagonNum ( n ) : NEW_LINE INDENT return ( 48 * n * n - 46 * n ) // 2 NEW_LINE DEDENT
n = 3 NEW_LINE print ( "3rd ▁ pentacontagon ▁ Number ▁ is ▁ = ▁ " , pentacontagonNum ( n ) ) NEW_LINE
from queue import PriorityQueue NEW_LINE
def lastElement ( arr ) : NEW_LINE
pq = PriorityQueue ( ) NEW_LINE for i in range ( len ( arr ) ) : NEW_LINE
pq . put ( - 1 * arr [ i ] ) NEW_LINE
m1 = 0 NEW_LINE m2 = 0 NEW_LINE
while not pq . empty ( ) : NEW_LINE
if pq . qsize ( ) == 1 : NEW_LINE
return - 1 * pq . get ( ) NEW_LINE else : NEW_LINE m1 = - 1 * pq . get ( ) NEW_LINE m2 = - 1 * pq . get ( ) NEW_LINE
if m1 != m2 : NEW_LINE INDENT pq . put ( - 1 * abs ( m1 - m2 ) ) NEW_LINE DEDENT return 0 NEW_LINE
arr = [ 2 , 7 , 4 , 1 , 8 , 1 , 1 ] NEW_LINE print ( lastElement ( arr ) ) NEW_LINE
import math NEW_LINE
def countDigit ( n ) : NEW_LINE INDENT return ( math . floor ( math . log10 ( n ) + 1 ) ) NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 80 NEW_LINE print ( countDigit ( n ) ) NEW_LINE DEDENT
def sum ( x , n ) : NEW_LINE INDENT total = 1.0 NEW_LINE multi = x NEW_LINE DEDENT
print ( 1 , end = " ▁ " ) NEW_LINE
for i in range ( 1 , n ) : NEW_LINE INDENT total = total + multi NEW_LINE print ( ' % .1f ' % multi , end = " ▁ " ) NEW_LINE multi = multi * x NEW_LINE DEDENT print ( ' ' ) NEW_LINE return total ; NEW_LINE
x = 2 NEW_LINE n = 5 NEW_LINE print ( ' % .2f ' % sum ( x , n ) ) NEW_LINE
def findRemainder ( n ) : NEW_LINE
x = n & 3 NEW_LINE
return x NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 43 NEW_LINE ans = findRemainder ( N ) NEW_LINE print ( ans ) NEW_LINE DEDENT
def triangular_series ( n ) : NEW_LINE INDENT j = 1 NEW_LINE k = 1 NEW_LINE DEDENT
for i in range ( 1 , n + 1 ) : NEW_LINE INDENT print ( k , end = ' ▁ ' ) NEW_LINE DEDENT
j = j + 1 NEW_LINE
k = k + j NEW_LINE
n = 5 NEW_LINE triangular_series ( n ) NEW_LINE
def countDigit ( n ) : NEW_LINE INDENT if n / 10 == 0 : NEW_LINE INDENT return 1 NEW_LINE DEDENT return 1 + countDigit ( n // 10 ) NEW_LINE DEDENT
n = 345289467 NEW_LINE print ( " Number ▁ of ▁ digits ▁ : ▁ % ▁ d " % ( countDigit ( n ) ) ) NEW_LINE
x = 1234 NEW_LINE
if ( x % 9 == 1 ) : NEW_LINE INDENT print ( " Magic ▁ Number " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Not ▁ a ▁ Magic ▁ Number " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT MAX = 100 NEW_LINE DEDENT
arr = [ 0 for i in range ( MAX ) ] NEW_LINE arr [ 0 ] = 0 NEW_LINE arr [ 1 ] = 1 NEW_LINE for i in range ( 2 , MAX ) : NEW_LINE INDENT arr [ i ] = arr [ i - 1 ] + arr [ i - 2 ] NEW_LINE DEDENT print ( " Fibonacci ▁ numbers ▁ divisible ▁ by ▁ their ▁ indexes ▁ are ▁ : " ) NEW_LINE for i in range ( 1 , MAX ) : NEW_LINE INDENT if ( arr [ i ] % i == 0 ) : NEW_LINE INDENT print ( i , end = " ▁ " ) NEW_LINE DEDENT DEDENT
import sys NEW_LINE def findMaxValue ( ) : NEW_LINE INDENT res = 2 ; NEW_LINE fact = 2 ; NEW_LINE while ( True ) : NEW_LINE DEDENT
if ( fact < 0 or fact > sys . maxsize ) : NEW_LINE INDENT break ; NEW_LINE DEDENT res += 1 ; NEW_LINE fact = fact * res ; NEW_LINE return res - 1 ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT print ( " Maximum ▁ value ▁ of ▁ integer : " , findMaxValue ( ) ) ; NEW_LINE DEDENT
import math NEW_LINE
def firstkdigits ( n , k ) : NEW_LINE
product = n * math . log ( n , 10 ) ; NEW_LINE
decimal_part = product - math . floor ( product ) ; NEW_LINE
decimal_part = pow ( 10 , decimal_part ) ; NEW_LINE
digits = pow ( 10 , k - 1 ) ; NEW_LINE return math . floor ( decimal_part * digits ) ; NEW_LINE
n = 1450 ; NEW_LINE k = 6 ; NEW_LINE print ( firstkdigits ( n , k ) ) ; NEW_LINE
def moduloMultiplication ( a , b , mod ) : NEW_LINE
a = a % mod ; NEW_LINE while ( b ) : NEW_LINE
if ( b & 1 ) : NEW_LINE INDENT res = ( res + a ) % mod ; NEW_LINE DEDENT
a = ( 2 * a ) % mod ; NEW_LINE
return res ; NEW_LINE
a = 10123465234878998 ; NEW_LINE b = 65746311545646431 ; NEW_LINE m = 10005412336548794 ; NEW_LINE print ( moduloMultiplication ( a , b , m ) ) ; NEW_LINE
import math NEW_LINE
def findRoots ( a , b , c ) : NEW_LINE
if a == 0 : NEW_LINE INDENT print ( " Invalid " ) NEW_LINE return - 1 NEW_LINE DEDENT d = b * b - 4 * a * c NEW_LINE sqrt_val = math . sqrt ( abs ( d ) ) NEW_LINE if d > 0 : NEW_LINE INDENT print ( " Roots ▁ are ▁ real ▁ and ▁ different ▁ " ) NEW_LINE print ( ( - b + sqrt_val ) / ( 2 * a ) ) NEW_LINE print ( ( - b - sqrt_val ) / ( 2 * a ) ) NEW_LINE DEDENT elif d == 0 : NEW_LINE INDENT print ( " Roots ▁ are ▁ real ▁ and ▁ same " ) NEW_LINE print ( - b / ( 2 * a ) ) NEW_LINE DEDENT
print ( " Roots ▁ are ▁ complex " ) NEW_LINE print ( - b / ( 2 * a ) , " ▁ + ▁ i " , sqrt_val ) NEW_LINE print ( - b / ( 2 * a ) , " ▁ - ▁ i " , sqrt_val ) NEW_LINE
a = 1 NEW_LINE b = - 7 NEW_LINE c = 12 NEW_LINE
findRoots ( a , b , c ) NEW_LINE
' NEW_LINE def val ( c ) : NEW_LINE INDENT if c >= '0' and c <= '9' : NEW_LINE INDENT return ord ( c ) - ord ( '0' ) NEW_LINE DEDENT else : NEW_LINE INDENT return ord ( c ) - ord ( ' A ' ) + 10 ; NEW_LINE DEDENT DEDENT
def toDeci ( str , base ) : NEW_LINE INDENT llen = len ( str ) NEW_LINE DEDENT
power = 1 NEW_LINE
num = 0 NEW_LINE
for i in range ( llen - 1 , - 1 , - 1 ) : NEW_LINE
if val ( str [ i ] ) >= base : NEW_LINE INDENT print ( ' Invalid ▁ Number ' ) NEW_LINE return - 1 NEW_LINE DEDENT num += val ( str [ i ] ) * power NEW_LINE power = power * base NEW_LINE return num NEW_LINE
strr = "11A " NEW_LINE base = 16 NEW_LINE print ( ' Decimal ▁ equivalent ▁ of ' , strr , ' in ▁ base ' , base , ' is ' , toDeci ( strr , base ) ) NEW_LINE
def seriesSum ( calculated , current , N ) : NEW_LINE INDENT i = calculated ; NEW_LINE cur = 1 ; NEW_LINE DEDENT
if ( current == N + 1 ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT
while ( i < calculated + current ) : NEW_LINE INDENT cur *= i ; NEW_LINE i += 1 ; NEW_LINE DEDENT
return cur + seriesSum ( i , current + 1 , N ) ; NEW_LINE
N = 5 ; NEW_LINE
print ( seriesSum ( 1 , 1 , N ) ) ; NEW_LINE
N = 30 NEW_LINE
fib = [ 0 for i in range ( N ) ] NEW_LINE
def largestFiboLessOrEqual ( n ) : NEW_LINE
fib [ 0 ] = 1 NEW_LINE
fib [ 1 ] = 2 NEW_LINE
i = 2 NEW_LINE while fib [ i - 1 ] <= n : NEW_LINE INDENT fib [ i ] = fib [ i - 1 ] + fib [ i - 2 ] NEW_LINE i += 1 NEW_LINE DEDENT
return ( i - 2 ) NEW_LINE
def fibonacciEncoding ( n ) : NEW_LINE INDENT index = largestFiboLessOrEqual ( n ) NEW_LINE DEDENT
codeword = [ ' a ' for i in range ( index + 2 ) ] NEW_LINE
i = index NEW_LINE while ( n ) : NEW_LINE
codeword [ i ] = '1' NEW_LINE
n = n - fib [ i ] NEW_LINE
i = i - 1 NEW_LINE
while ( i >= 0 and fib [ i ] > n ) : NEW_LINE INDENT codeword [ i ] = '0' NEW_LINE i = i - 1 NEW_LINE DEDENT
codeword [ index + 1 ] = '1' NEW_LINE
return " " . join ( codeword ) NEW_LINE
n = 143 NEW_LINE print ( " Fibonacci ▁ code ▁ word ▁ for " , n , " is " , fibonacciEncoding ( n ) ) NEW_LINE
def countSquares ( m , n ) : NEW_LINE
if ( n < m ) : NEW_LINE INDENT temp = m NEW_LINE m = n NEW_LINE n = temp NEW_LINE DEDENT
return n * ( n + 1 ) * ( 3 * m - n + 1 ) // 6 NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT m = 4 NEW_LINE n = 3 NEW_LINE print ( " Count ▁ of ▁ squares ▁ is " , countSquares ( m , n ) ) NEW_LINE DEDENT
def simpleSieve ( limit ) : NEW_LINE
mark = [ True for i in range ( limit ) ] NEW_LINE
for p in range ( p * p , limit - 1 , 1 ) : NEW_LINE
if ( mark [ p ] == True ) : NEW_LINE
for i in range ( p * p , limit - 1 , p ) : NEW_LINE INDENT mark [ i ] = False NEW_LINE DEDENT
for p in range ( 2 , limit - 1 , 1 ) : NEW_LINE INDENT if ( mark [ p ] == True ) : NEW_LINE INDENT print ( p , end = " ▁ " ) NEW_LINE DEDENT DEDENT
def modInverse ( a , m ) : NEW_LINE INDENT m0 = m NEW_LINE y = 0 NEW_LINE x = 1 NEW_LINE if ( m == 1 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT while ( a > 1 ) : NEW_LINE DEDENT
q = a // m NEW_LINE t = m NEW_LINE
m = a % m NEW_LINE a = t NEW_LINE t = y NEW_LINE
y = x - q * y NEW_LINE x = t NEW_LINE
if ( x < 0 ) : NEW_LINE INDENT x = x + m0 NEW_LINE DEDENT return x NEW_LINE
a = 3 NEW_LINE m = 11 NEW_LINE
print ( " Modular ▁ multiplicative ▁ inverse ▁ is " , modInverse ( a , m ) ) NEW_LINE
def gcd ( a , b ) : NEW_LINE INDENT if ( a == 0 ) : NEW_LINE INDENT return b NEW_LINE DEDENT return gcd ( b % a , a ) NEW_LINE DEDENT
def phi ( n ) : NEW_LINE INDENT result = 1 NEW_LINE for i in range ( 2 , n ) : NEW_LINE INDENT if ( gcd ( i , n ) == 1 ) : NEW_LINE INDENT result += 1 NEW_LINE DEDENT DEDENT return result NEW_LINE DEDENT
for n in range ( 1 , 11 ) : NEW_LINE INDENT print ( " phi ( " , n , " ) ▁ = ▁ " , phi ( n ) , sep = " " ) NEW_LINE DEDENT
def phi ( n ) : NEW_LINE
p = 2 NEW_LINE while p * p <= n : NEW_LINE
if n % p == 0 : NEW_LINE
while n % p == 0 : NEW_LINE INDENT n = n // p NEW_LINE DEDENT result = result * ( 1.0 - ( 1.0 / float ( p ) ) ) NEW_LINE p = p + 1 NEW_LINE
if n > 1 : NEW_LINE INDENT result = result * ( 1.0 - ( 1.0 / float ( n ) ) ) NEW_LINE DEDENT return int ( result ) NEW_LINE
for n in range ( 1 , 11 ) : NEW_LINE INDENT print ( " phi ( " , n , " ) ▁ = ▁ " , phi ( n ) ) NEW_LINE DEDENT
def printFibonacciNumbers ( n ) : NEW_LINE INDENT f1 = 0 NEW_LINE f2 = 1 NEW_LINE if ( n < 1 ) : NEW_LINE INDENT return NEW_LINE DEDENT print ( f1 , end = " ▁ " ) NEW_LINE for x in range ( 1 , n ) : NEW_LINE INDENT print ( f2 , end = " ▁ " ) NEW_LINE next = f1 + f2 NEW_LINE f1 = f2 NEW_LINE f2 = next NEW_LINE DEDENT DEDENT
printFibonacciNumbers ( 7 ) NEW_LINE
def gcd ( a , b ) : NEW_LINE INDENT if a == 0 : NEW_LINE INDENT return b NEW_LINE DEDENT return gcd ( b % a , a ) NEW_LINE DEDENT
def lcm ( a , b ) : NEW_LINE INDENT return ( a / gcd ( a , b ) ) * b NEW_LINE DEDENT
a = 15 NEW_LINE b = 20 NEW_LINE print ( ' LCM ▁ of ' , a , ' and ' , b , ' is ' , lcm ( a , b ) ) NEW_LINE
def convert_to_words ( num ) : NEW_LINE
if ( l == 0 ) : NEW_LINE INDENT print ( " empty ▁ string " ) NEW_LINE return NEW_LINE DEDENT if ( l > 4 ) : NEW_LINE INDENT print ( " Length ▁ more ▁ than ▁ 4 ▁ is ▁ not ▁ supported " ) NEW_LINE return NEW_LINE DEDENT
single_digits = [ " zero " , " one " , " two " , " three " , " four " , " five " , " six " , " seven " , " eight " , " nine " ] NEW_LINE
two_digits = [ " " , " ten " , " eleven " , " twelve " , " thirteen " , " fourteen " , " fifteen " , " sixteen " , " seventeen " , " eighteen " , " nineteen " ] NEW_LINE
tens_multiple = [ " " , " " , " twenty " , " thirty " , " forty " , " fifty " , " sixty " , " seventy " , " eighty " , " ninety " ] NEW_LINE tens_power = [ " hundred " , " thousand " ] NEW_LINE
print ( num , " : " , end = " ▁ " ) NEW_LINE
if ( l == 1 ) : NEW_LINE INDENT print ( single_digits [ ord ( num [ 0 ] ) - 48 ] ) NEW_LINE return NEW_LINE DEDENT
' NEW_LINE INDENT x = 0 NEW_LINE while ( x < len ( num ) ) : NEW_LINE DEDENT
if ( l >= 3 ) : NEW_LINE INDENT if ( ord ( num [ x ] ) - 48 != 0 ) : NEW_LINE INDENT print ( single_digits [ ord ( num [ x ] ) - 48 ] , end = " ▁ " ) NEW_LINE print ( tens_power [ l - 3 ] , end = " ▁ " ) NEW_LINE DEDENT DEDENT
l -= 1 NEW_LINE
else : NEW_LINE
if ( ord ( num [ x ] ) - 48 == 1 ) : NEW_LINE INDENT sum = ( ord ( num [ x ] ) - 48 + ord ( num [ x + 1 ] ) - 48 ) NEW_LINE print ( two_digits [ sum ] ) NEW_LINE return NEW_LINE DEDENT
elif ( ord ( num [ x ] ) - 48 == 2 and ord ( num [ x + 1 ] ) - 48 == 0 ) : NEW_LINE INDENT print ( " twenty " ) NEW_LINE return NEW_LINE DEDENT
else : NEW_LINE INDENT i = ord ( num [ x ] ) - 48 NEW_LINE if ( i > 0 ) : NEW_LINE INDENT print ( tens_multiple [ i ] , end = " ▁ " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " " , end = " " ) NEW_LINE DEDENT x += 1 NEW_LINE if ( ord ( num [ x ] ) - 48 != 0 ) : NEW_LINE INDENT print ( single_digits [ ord ( num [ x ] ) - 48 ] ) NEW_LINE DEDENT DEDENT x += 1 NEW_LINE
MAX = 11 ; NEW_LINE def isMultipleof5 ( n ) : NEW_LINE INDENT s = str ( n ) ; NEW_LINE l = len ( s ) ; NEW_LINE DEDENT
if ( s [ l - 1 ] == '5' or s [ l - 1 ] == '0' ) : NEW_LINE INDENT return True ; NEW_LINE DEDENT return False ; NEW_LINE
n = 19 ; NEW_LINE if ( isMultipleof5 ( n ) == True ) : NEW_LINE INDENT print ( n , " is ▁ multiple ▁ of ▁ 5" ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( n , " is ▁ not ▁ a ▁ multiple ▁ of ▁ 5" ) ; NEW_LINE DEDENT
def toggleBit ( n , k ) : NEW_LINE INDENT return ( n ^ ( 1 << ( k - 1 ) ) ) ; NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 5 ; k = 2 ; NEW_LINE print ( toggleBit ( n , k ) ) ; NEW_LINE DEDENT
def clearBit ( n , k ) : NEW_LINE INDENT return ( n & ( ~ ( 1 << ( k - 1 ) ) ) ) NEW_LINE DEDENT
n = 5 NEW_LINE k = 1 NEW_LINE print ( clearBit ( n , k ) ) NEW_LINE
def add ( x , y ) : NEW_LINE INDENT keep = ( x & y ) << 1 ; NEW_LINE res = x ^ y ; NEW_LINE DEDENT
if ( keep == 0 ) : NEW_LINE INDENT return res ; NEW_LINE DEDENT return add ( keep , res ) ; NEW_LINE
print ( add ( 15 , 38 ) ) ; NEW_LINE
import math NEW_LINE def countBits ( number ) : NEW_LINE
return int ( ( math . log ( number ) / math . log ( 2 ) ) + 1 ) ; NEW_LINE
num = 65 ; NEW_LINE print ( countBits ( num ) ) ; NEW_LINE
INT_SIZE = 32 NEW_LINE
def constructNthNumber ( group_no , aux_num , op ) : NEW_LINE INDENT a = [ 0 ] * INT_SIZE NEW_LINE num , i = 0 , 0 NEW_LINE DEDENT
if op == 2 : NEW_LINE
len_f = 2 * group_no NEW_LINE
a [ len_f - 1 ] = a [ 0 ] = 1 NEW_LINE
while aux_num : NEW_LINE
a [ group_no + i ] = a [ group_no - 1 - i ] = aux_num & 1 NEW_LINE aux_num = aux_num >> 1 NEW_LINE i += 1 NEW_LINE
elif op == 0 : NEW_LINE
len_f = 2 * group_no + 1 NEW_LINE
a [ len_f - 1 ] = a [ 0 ] = 1 NEW_LINE a [ group_no ] = 0 NEW_LINE
while aux_num : NEW_LINE
a [ group_no + 1 + i ] = a [ group_no - 1 - i ] = aux_num & 1 NEW_LINE aux_num = aux_num >> 1 NEW_LINE i += 1 NEW_LINE
len_f = 2 * group_no + 1 NEW_LINE
a [ len_f - 1 ] = a [ 0 ] = 1 NEW_LINE a [ group_no ] = 1 NEW_LINE
while aux_num : NEW_LINE
a [ group_no + 1 + i ] = a [ group_no - 1 - i ] = aux_num & 1 NEW_LINE aux_num = aux_num >> 1 NEW_LINE i += 1 NEW_LINE
for i in range ( 0 , len_f ) : NEW_LINE INDENT num += ( 1 << i ) * a [ i ] NEW_LINE DEDENT return num NEW_LINE
def getNthNumber ( n ) : NEW_LINE INDENT group_no = 0 NEW_LINE count_upto_group , count_temp = 0 , 1 NEW_LINE DEDENT
while count_temp < n : NEW_LINE INDENT group_no += 1 NEW_LINE DEDENT
count_upto_group = count_temp NEW_LINE count_temp += 3 * ( 1 << ( group_no - 1 ) ) NEW_LINE
group_offset = n - count_upto_group - 1 NEW_LINE
if ( group_offset + 1 ) <= ( 1 << ( group_no - 1 ) ) : NEW_LINE
aux_num = group_offset NEW_LINE else : NEW_LINE if ( ( ( group_offset + 1 ) - ( 1 << ( group_no - 1 ) ) ) % 2 ) : NEW_LINE
else : NEW_LINE
aux_num = ( ( ( group_offset ) - ( 1 << ( group_no - 1 ) ) ) // 2 ) NEW_LINE return constructNthNumber ( group_no , aux_num , op ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 9 NEW_LINE DEDENT
print ( getNthNumber ( n ) ) NEW_LINE
def toggleAllExceptK ( n , k ) : NEW_LINE
temp = bin ( n ^ ( 1 << k ) ) [ 2 : ] NEW_LINE ans = " " NEW_LINE for i in temp : NEW_LINE INDENT if i == '1' : NEW_LINE INDENT ans += '0' NEW_LINE DEDENT else : NEW_LINE INDENT ans += '1' NEW_LINE DEDENT DEDENT return int ( ans , 2 ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 4294967295 NEW_LINE k = 0 NEW_LINE print ( toggleAllExceptK ( n , k ) ) NEW_LINE DEDENT
def swapBits ( n , p1 , p2 ) : NEW_LINE
bit1 = ( n >> p1 ) & 1 NEW_LINE
bit2 = ( n >> p2 ) & 1 NEW_LINE
x = ( bit1 ^ bit2 ) NEW_LINE
x = ( x << p1 ) | ( x << p2 ) NEW_LINE
result = n ^ x NEW_LINE return result NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT res = swapBits ( 28 , 0 , 3 ) NEW_LINE print ( " Result ▁ = ▁ " , res ) NEW_LINE DEDENT
def invert ( x ) : NEW_LINE INDENT if ( x == 1 ) : NEW_LINE INDENT return 2 NEW_LINE else : NEW_LINE return 1 NEW_LINE DEDENT DEDENT
def invertSub ( x ) : NEW_LINE INDENT return ( 3 - x ) NEW_LINE DEDENT
import math as mt NEW_LINE NO_OF_CHARS = 256 NEW_LINE
def firstNonRepeating ( string ) : NEW_LINE
arr = [ - 1 for i in range ( NO_OF_CHARS ) ] NEW_LINE
for i in range ( len ( string ) ) : NEW_LINE INDENT if arr [ ord ( string [ i ] ) ] == - 1 : NEW_LINE INDENT arr [ ord ( string [ i ] ) ] = i NEW_LINE DEDENT else : NEW_LINE INDENT arr [ ord ( string [ i ] ) ] = - 2 NEW_LINE DEDENT DEDENT res = 10 ** 18 NEW_LINE for i in range ( NO_OF_CHARS ) : NEW_LINE
if arr [ i ] >= 0 : NEW_LINE INDENT res = min ( res , arr [ i ] ) NEW_LINE DEDENT return res NEW_LINE
string = " geeksforgeeks " NEW_LINE index = firstNonRepeating ( string ) NEW_LINE if index == 10 ** 18 : NEW_LINE INDENT print ( " Either ▁ all ▁ characters ▁ are ▁ repeating ▁ or ▁ string ▁ is ▁ empty " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " First ▁ non - repeating ▁ character ▁ is " , string [ index ] ) NEW_LINE DEDENT
def triacontagonalNum ( n ) : NEW_LINE INDENT return ( 28 * n * n - 26 * n ) // 2 NEW_LINE DEDENT
n = 3 NEW_LINE print ( "3rd ▁ triacontagonal ▁ Number ▁ is ▁ = ▁ " , triacontagonalNum ( n ) ) NEW_LINE
def hexacontagonNum ( n ) : NEW_LINE INDENT return ( 58 * n * n - 56 * n ) // 2 NEW_LINE DEDENT
n = 3 NEW_LINE print ( "3rd ▁ hexacontagon ▁ Number ▁ is ▁ = ▁ " , hexacontagonNum ( n ) ) ; NEW_LINE
def enneacontagonNum ( n ) : NEW_LINE INDENT return ( 88 * n * n - 86 * n ) // 2 NEW_LINE DEDENT
n = 3 NEW_LINE print ( "3rd ▁ enneacontagon ▁ Number ▁ is ▁ = ▁ " , enneacontagonNum ( n ) ) NEW_LINE
def triacontakaidigonNum ( n ) : NEW_LINE INDENT return ( 30 * n * n - 28 * n ) // 2 NEW_LINE DEDENT
n = 3 NEW_LINE print ( "3rd ▁ triacontakaidigon ▁ Number ▁ is ▁ = ▁ " , triacontakaidigonNum ( n ) ) NEW_LINE
def IcosihexagonalNum ( n ) : NEW_LINE INDENT return ( 24 * n * n - 22 * n ) // 2 NEW_LINE DEDENT
n = 3 NEW_LINE print ( "3rd ▁ Icosihexagonal ▁ Number ▁ is ▁ = ▁ " , IcosihexagonalNum ( n ) ) NEW_LINE
def icosikaioctagonalNum ( n ) : NEW_LINE INDENT return ( 26 * n * n - 24 * n ) // 2 NEW_LINE DEDENT
n = 3 NEW_LINE print ( "3rd ▁ icosikaioctagonal ▁ Number ▁ is ▁ = ▁ " , icosikaioctagonalNum ( n ) ) NEW_LINE
def octacontagonNum ( n ) : NEW_LINE INDENT return ( 78 * n * n - 76 * n ) // 2 NEW_LINE DEDENT
n = 3 NEW_LINE print ( "3rd ▁ octacontagon ▁ Number ▁ is ▁ = ▁ " , octacontagonNum ( n ) ) NEW_LINE
def hectagonNum ( n ) : NEW_LINE INDENT return ( 98 * n * n - 96 * n ) // 2 NEW_LINE DEDENT
n = 3 NEW_LINE print ( "3rd ▁ hectagon ▁ Number ▁ is ▁ = ▁ " , hectagonNum ( n ) ) NEW_LINE
def tetracontagonNum ( n ) : NEW_LINE INDENT return ( 38 * n * n - 36 * n ) // 2 NEW_LINE DEDENT
n = 3 NEW_LINE print ( "3rd ▁ tetracontagon ▁ Number ▁ is ▁ = ▁ " , tetracontagonNum ( n ) ) NEW_LINE
def binarySearch ( arr , N , X ) : NEW_LINE
start = 0 NEW_LINE
end = N NEW_LINE while ( start <= end ) : NEW_LINE
mid = start + ( end - start ) // 2 NEW_LINE
if ( X == arr [ mid ] ) : NEW_LINE
return mid NEW_LINE
elif ( X < arr [ mid ] ) : NEW_LINE
start = mid + 1 NEW_LINE else : NEW_LINE
end = mid - 1 NEW_LINE
return - 1 NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 5 , 4 , 3 , 2 , 1 ] NEW_LINE N = len ( arr ) NEW_LINE X = 5 NEW_LINE print ( binarySearch ( arr , N , X ) ) NEW_LINE DEDENT
def flip ( arr , i ) : NEW_LINE INDENT start = 0 NEW_LINE while start < i : NEW_LINE INDENT temp = arr [ start ] NEW_LINE arr [ start ] = arr [ i ] NEW_LINE arr [ i ] = temp NEW_LINE start += 1 NEW_LINE i -= 1 NEW_LINE DEDENT DEDENT
def findMax ( arr , n ) : NEW_LINE INDENT mi = 0 NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT if arr [ i ] > arr [ mi ] : NEW_LINE INDENT mi = i NEW_LINE DEDENT DEDENT return mi NEW_LINE DEDENT
def pancakeSort ( arr , n ) : NEW_LINE
curr_size = n NEW_LINE while curr_size > 1 : NEW_LINE
mi = findMax ( arr , curr_size ) NEW_LINE
if mi != curr_size - 1 : NEW_LINE
flip ( arr , mi ) NEW_LINE
flip ( arr , curr_size - 1 ) NEW_LINE curr_size -= 1 NEW_LINE
def printArray ( arr , n ) : NEW_LINE INDENT for i in range ( 0 , n ) : NEW_LINE INDENT print ( " % d " % ( arr [ i ] ) , end = " ▁ " ) NEW_LINE DEDENT DEDENT
arr = [ 23 , 10 , 20 , 11 , 12 , 6 , 7 ] NEW_LINE n = len ( arr ) NEW_LINE pancakeSort ( arr , n ) ; NEW_LINE print ( " Sorted ▁ Array ▁ " ) NEW_LINE printArray ( arr , n ) NEW_LINE
