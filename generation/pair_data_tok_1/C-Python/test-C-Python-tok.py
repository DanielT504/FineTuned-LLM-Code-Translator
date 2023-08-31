import math NEW_LINE
def distance ( a1 , b1 , c1 , d1 , a2 , b2 , c2 , d2 ) : NEW_LINE INDENT if ( a1 / a2 == b1 / b2 and b1 / b2 == c1 / c2 ) : NEW_LINE INDENT x1 = y1 = 0 NEW_LINE z1 = - d1 / c1 NEW_LINE d = abs ( ( c2 * z1 + d2 ) ) / ( math . sqrt ( a2 * a2 + b2 * b2 + c2 * c2 ) ) NEW_LINE print ( " Perpendicular ▁ distance ▁ is " ) , d NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Planes ▁ are ▁ not ▁ parallel " ) NEW_LINE DEDENT DEDENT
a1 = 1 NEW_LINE b1 = 2 NEW_LINE c1 = - 1 NEW_LINE d1 = 1 NEW_LINE a2 = 3 NEW_LINE b2 = 6 NEW_LINE c2 = - 3 NEW_LINE d2 = - 4 NEW_LINE distance ( a1 , b1 , c1 , d1 , a2 , b2 , c2 , d2 ) NEW_LINE
def Series ( n ) : NEW_LINE INDENT sums = 0 NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE INDENT sums += ( i * i ) ; NEW_LINE DEDENT return sums NEW_LINE DEDENT
n = 3 NEW_LINE res = Series ( n ) NEW_LINE print ( res ) NEW_LINE
def areElementsContiguous ( arr , n ) : NEW_LINE
arr . sort ( ) NEW_LINE
for i in range ( 1 , n ) : NEW_LINE INDENT if ( arr [ i ] - arr [ i - 1 ] > 1 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT DEDENT return 1 NEW_LINE
arr = [ 5 , 2 , 3 , 6 , 4 , 4 , 6 , 6 ] NEW_LINE n = len ( arr ) NEW_LINE if areElementsContiguous ( arr , n ) : print ( " Yes " ) NEW_LINE else : print ( " No " ) NEW_LINE
def leftRotatebyOne ( arr , n ) : NEW_LINE INDENT temp = arr [ 0 ] NEW_LINE for i in range ( n - 1 ) : NEW_LINE INDENT arr [ i ] = arr [ i + 1 ] NEW_LINE DEDENT arr [ n - 1 ] = temp NEW_LINE DEDENT
def leftRotate ( arr , d , n ) : NEW_LINE INDENT for i in range ( d ) : NEW_LINE INDENT leftRotatebyOne ( arr , n ) NEW_LINE DEDENT DEDENT
def printArray ( arr , size ) : NEW_LINE INDENT for i in range ( size ) : NEW_LINE INDENT print ( " % ▁ d " % arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT DEDENT
arr = [ 1 , 2 , 3 , 4 , 5 , 6 , 7 ] NEW_LINE leftRotate ( arr , 2 , 7 ) NEW_LINE printArray ( arr , 7 ) NEW_LINE
def findFirstMissing ( array , start , end ) : NEW_LINE INDENT if ( start > end ) : NEW_LINE INDENT return end + 1 NEW_LINE DEDENT if ( start != array [ start ] ) : NEW_LINE INDENT return start ; NEW_LINE DEDENT mid = int ( ( start + end ) / 2 ) NEW_LINE DEDENT
if ( array [ mid ] == mid ) : NEW_LINE INDENT return findFirstMissing ( array , mid + 1 , end ) NEW_LINE DEDENT return findFirstMissing ( array , start , mid ) NEW_LINE
arr = [ 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 10 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Smallest ▁ missing ▁ element ▁ is " , findFirstMissing ( arr , 0 , n - 1 ) ) NEW_LINE
def find_max_sum ( arr ) : NEW_LINE INDENT incl = 0 NEW_LINE excl = 0 NEW_LINE for i in arr : NEW_LINE DEDENT
new_excl = excl if excl > incl else incl NEW_LINE
incl = excl + i NEW_LINE excl = new_excl NEW_LINE
return ( excl if excl > incl else incl ) NEW_LINE
arr = [ 5 , 5 , 10 , 100 , 10 , 5 ] NEW_LINE print find_max_sum ( arr ) NEW_LINE
def isMajority ( arr , n , x ) : NEW_LINE
last_index = ( n // 2 + 1 ) if n % 2 == 0 else ( n // 2 ) NEW_LINE
for i in range ( last_index ) : NEW_LINE
if arr [ i ] == x and arr [ i + n // 2 ] == x : NEW_LINE INDENT return 1 NEW_LINE DEDENT
arr = [ 1 , 2 , 3 , 4 , 4 , 4 , 4 ] NEW_LINE n = len ( arr ) NEW_LINE x = 4 NEW_LINE if ( isMajority ( arr , n , x ) ) : NEW_LINE INDENT print ( " % ▁ d ▁ appears ▁ more ▁ than ▁ % ▁ d ▁ times ▁ in ▁ arr [ ] " % ( x , n // 2 ) ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " % ▁ d ▁ does ▁ not ▁ appear ▁ more ▁ than ▁ % ▁ d ▁ times ▁ in ▁ arr [ ] " % ( x , n // 2 ) ) NEW_LINE DEDENT
def _binarySearch ( arr , low , high , x ) : NEW_LINE INDENT if high >= low : NEW_LINE INDENT mid = ( low + high ) // 2 NEW_LINE DEDENT DEDENT
if ( mid == 0 or x > arr [ mid - 1 ] ) and ( arr [ mid ] == x ) : NEW_LINE INDENT return mid NEW_LINE DEDENT elif x > arr [ mid ] : NEW_LINE INDENT return _binarySearch ( arr , ( mid + 1 ) , high , x ) NEW_LINE DEDENT else : NEW_LINE INDENT return _binarySearch ( arr , low , ( mid - 1 ) , x ) NEW_LINE DEDENT return - 1 NEW_LINE
def isMajority ( arr , n , x ) : NEW_LINE
i = _binarySearch ( arr , 0 , n - 1 , x ) NEW_LINE
if i == - 1 : NEW_LINE INDENT return False NEW_LINE DEDENT
if ( ( i + n // 2 ) <= ( n - 1 ) ) and arr [ i + n // 2 ] == x : NEW_LINE INDENT return True NEW_LINE DEDENT else : NEW_LINE INDENT return False NEW_LINE DEDENT
def isMajorityElement ( arr , n , key ) : NEW_LINE INDENT if ( arr [ n // 2 ] == key ) : NEW_LINE INDENT return True NEW_LINE DEDENT return False NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 1 , 2 , 3 , 3 , 3 , 3 , 10 ] NEW_LINE n = len ( arr ) NEW_LINE x = 3 NEW_LINE if ( isMajorityElement ( arr , n , x ) ) : NEW_LINE INDENT print ( x , " ▁ appears ▁ more ▁ than ▁ " , n // 2 , " ▁ times ▁ in ▁ arr [ ] " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( x , " ▁ does ▁ not ▁ appear ▁ more ▁ than " , n // 2 , " ▁ times ▁ in ▁ arr [ ] " ) NEW_LINE DEDENT DEDENT
INT_MIN = - 32767 NEW_LINE
def cutRod ( price , n ) : NEW_LINE INDENT val = [ 0 for x in range ( n + 1 ) ] NEW_LINE val [ 0 ] = 0 NEW_LINE DEDENT
for i in range ( 1 , n + 1 ) : NEW_LINE INDENT max_val = INT_MIN NEW_LINE for j in range ( i ) : NEW_LINE INDENT max_val = max ( max_val , price [ j ] + val [ i - j - 1 ] ) NEW_LINE DEDENT val [ i ] = max_val NEW_LINE DEDENT return val [ n ] NEW_LINE
arr = [ 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 ] NEW_LINE size = len ( arr ) NEW_LINE print ( " Maximum ▁ Obtainable ▁ Value ▁ is ▁ " + str ( cutRod ( arr , size ) ) ) NEW_LINE
def Convert ( radian ) : NEW_LINE INDENT pi = 3.14159 NEW_LINE degree = radian * ( 180 / pi ) NEW_LINE return degree NEW_LINE DEDENT
radian = 5 NEW_LINE print ( " degree ▁ = " , ( Convert ( radian ) ) ) NEW_LINE
def subtract ( x , y ) : NEW_LINE
while ( y != 0 ) : NEW_LINE
borrow = ( ~ x ) & y NEW_LINE
x = x ^ y NEW_LINE
y = borrow << 1 NEW_LINE return x NEW_LINE
x = 29 NEW_LINE y = 13 NEW_LINE print ( " x ▁ - ▁ y ▁ is " , subtract ( x , y ) ) NEW_LINE
def subtract ( x , y ) : NEW_LINE INDENT if ( y == 0 ) : NEW_LINE INDENT return x NEW_LINE DEDENT return subtract ( x ^ y , ( ~ x & y ) << 1 ) NEW_LINE DEDENT
x = 29 NEW_LINE y = 13 NEW_LINE print ( " x ▁ - ▁ y ▁ is " , subtract ( x , y ) ) NEW_LINE
def reverse ( string ) : NEW_LINE INDENT if len ( string ) == 0 : NEW_LINE INDENT return NEW_LINE DEDENT temp = string [ 0 ] NEW_LINE reverse ( string [ 1 : ] ) NEW_LINE print ( temp , end = ' ' ) NEW_LINE DEDENT
string = " Geeks ▁ for ▁ Geeks " NEW_LINE reverse ( string ) NEW_LINE
cola = 2 NEW_LINE rowa = 3 NEW_LINE colb = 3 NEW_LINE rowb = 2 NEW_LINE
def Kroneckerproduct ( A , B ) : NEW_LINE INDENT C = [ [ 0 for j in range ( cola * colb ) ] for i in range ( rowa * rowb ) ] NEW_LINE DEDENT
for i in range ( 0 , rowa ) : NEW_LINE
for k in range ( 0 , rowb ) : NEW_LINE
for j in range ( 0 , cola ) : NEW_LINE
for l in range ( 0 , colb ) : NEW_LINE
C [ i + l + 1 ] [ j + k + 1 ] = A [ i ] [ j ] * B [ k ] [ l ] NEW_LINE print ( C [ i + l + 1 ] [ j + k + 1 ] , end = ' ▁ ' ) NEW_LINE print ( " " ) NEW_LINE
A = [ [ 0 for j in range ( 2 ) ] for i in range ( 3 ) ] NEW_LINE B = [ [ 0 for j in range ( 3 ) ] for i in range ( 2 ) ] NEW_LINE A [ 0 ] [ 0 ] = 1 NEW_LINE A [ 0 ] [ 1 ] = 2 NEW_LINE A [ 1 ] [ 0 ] = 3 NEW_LINE A [ 1 ] [ 1 ] = 4 NEW_LINE A [ 2 ] [ 0 ] = 1 NEW_LINE A [ 2 ] [ 1 ] = 0 NEW_LINE B [ 0 ] [ 0 ] = 0 NEW_LINE B [ 0 ] [ 1 ] = 5 NEW_LINE B [ 0 ] [ 2 ] = 2 NEW_LINE B [ 1 ] [ 0 ] = 6 NEW_LINE B [ 1 ] [ 1 ] = 7 NEW_LINE B [ 1 ] [ 2 ] = 3 NEW_LINE Kroneckerproduct ( A , B ) NEW_LINE
import sys NEW_LINE
def MatrixChainOrder ( p , n ) : NEW_LINE
m = [ [ 0 for x in range ( n ) ] for x in range ( n ) ] NEW_LINE
for i in range ( 1 , n ) : NEW_LINE INDENT m [ i ] [ i ] = 0 NEW_LINE DEDENT
for L in range ( 2 , n ) : NEW_LINE INDENT for i in range ( 1 , n - L + 1 ) : NEW_LINE INDENT j = i + L - 1 NEW_LINE m [ i ] [ j ] = sys . maxint NEW_LINE for k in range ( i , j ) : NEW_LINE DEDENT DEDENT
q = m [ i ] [ k ] + m [ k + 1 ] [ j ] + p [ i - 1 ] * p [ k ] * p [ j ] NEW_LINE if q < m [ i ] [ j ] : NEW_LINE INDENT m [ i ] [ j ] = q NEW_LINE DEDENT return m [ 1 ] [ n - 1 ] NEW_LINE
arr = [ 1 , 2 , 3 , 4 ] NEW_LINE size = len ( arr ) NEW_LINE print ( " Minimum ▁ number ▁ of ▁ multiplications ▁ is ▁ " + str ( MatrixChainOrder ( arr , size ) ) ) NEW_LINE
def multiply ( x , y ) : NEW_LINE
if ( y == 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
if ( y > 0 ) : NEW_LINE INDENT return ( x + multiply ( x , y - 1 ) ) NEW_LINE DEDENT
if ( y < 0 ) : NEW_LINE INDENT return - multiply ( x , - y ) NEW_LINE DEDENT
print ( multiply ( 5 , - 11 ) ) NEW_LINE
def printPascal ( n : int ) : NEW_LINE
arr = [ [ 0 for x in range ( n ) ] for y in range ( n ) ] NEW_LINE
for line in range ( 0 , n ) : NEW_LINE
for i in range ( 0 , line + 1 ) : NEW_LINE
if ( i == 0 or i == line ) : NEW_LINE INDENT arr [ line ] [ i ] = 1 NEW_LINE print ( arr [ line ] [ i ] , end = " ▁ " ) NEW_LINE DEDENT
else : NEW_LINE INDENT arr [ line ] [ i ] = ( arr [ line - 1 ] [ i - 1 ] + arr [ line - 1 ] [ i ] ) NEW_LINE print ( arr [ line ] [ i ] , end = " ▁ " ) NEW_LINE DEDENT print ( " " , ▁ end ▁ = ▁ " " ) NEW_LINE
n = 5 NEW_LINE printPascal ( n ) NEW_LINE
def printPascal ( n ) : NEW_LINE INDENT for line in range ( 1 , n + 1 ) : NEW_LINE DEDENT
C = 1 ; NEW_LINE for i in range ( 1 , line + 1 ) : NEW_LINE
print ( C , end = " ▁ " ) ; NEW_LINE C = int ( C * ( line - i ) / i ) ; NEW_LINE print ( " " ) ; NEW_LINE
n = 5 ; NEW_LINE printPascal ( n ) ; NEW_LINE
def Add ( x , y ) : NEW_LINE
while ( y != 0 ) : NEW_LINE
carry = x & y NEW_LINE
x = x ^ y NEW_LINE
y = carry << 1 NEW_LINE return x NEW_LINE
print ( Add ( 15 , 32 ) ) NEW_LINE
def Add ( x , y ) : NEW_LINE INDENT if ( y == 0 ) : NEW_LINE INDENT return x NEW_LINE DEDENT else : NEW_LINE INDENT return Add ( x ^ y , ( x & y ) << 1 ) NEW_LINE DEDENT DEDENT
def countSetBits ( n ) : NEW_LINE INDENT count = 0 NEW_LINE while ( n ) : NEW_LINE INDENT count += n & 1 NEW_LINE n >>= 1 NEW_LINE DEDENT return count NEW_LINE DEDENT
i = 9 NEW_LINE print ( countSetBits ( i ) ) NEW_LINE
num_to_bits = [ 0 , 1 , 1 , 2 , 1 , 2 , 2 , 3 , 1 , 2 , 2 , 3 , 2 , 3 , 3 , 4 ] ; NEW_LINE
def countSetBitsRec ( num ) : NEW_LINE INDENT nibble = 0 ; NEW_LINE if ( 0 == num ) : NEW_LINE INDENT return num_to_bits [ 0 ] ; NEW_LINE DEDENT DEDENT
nibble = num & 0xf ; NEW_LINE
return num_to_bits [ nibble ] + countSetBitsRec ( num >> 4 ) ; NEW_LINE
num = 31 ; NEW_LINE print ( countSetBitsRec ( num ) ) ; NEW_LINE
def countSetBits ( N ) : NEW_LINE INDENT count = 0 NEW_LINE DEDENT
INDENT for i in range ( 4 * 8 ) : NEW_LINE INDENT if ( N & ( 1 << i ) ) : NEW_LINE count += 1 NEW_LINE return count NEW_LINE DEDENT DEDENT
N = 15 NEW_LINE print ( countSetBits ( N ) ) NEW_LINE
def getParity ( n ) : NEW_LINE INDENT parity = 0 NEW_LINE while n : NEW_LINE INDENT parity = ~ parity NEW_LINE n = n & ( n - 1 ) NEW_LINE DEDENT return parity NEW_LINE DEDENT
n = 7 NEW_LINE print ( " Parity ▁ of ▁ no ▁ " , n , " ▁ = ▁ " , ( " odd " if getParity ( n ) else " even " ) ) NEW_LINE
import math NEW_LINE
def isPowerOfTwo ( n ) : NEW_LINE INDENT return ( math . ceil ( Log2 ( n ) ) == math . floor ( Log2 ( n ) ) ) ; NEW_LINE DEDENT
if ( isPowerOfTwo ( 31 ) ) : NEW_LINE INDENT print ( " Yes " ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) ; NEW_LINE DEDENT if ( isPowerOfTwo ( 64 ) ) : NEW_LINE INDENT print ( " Yes " ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) ; NEW_LINE DEDENT
def isPowerOfTwo ( n ) : NEW_LINE INDENT if ( n == 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT while ( n != 1 ) : NEW_LINE INDENT if ( n % 2 != 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT n = n // 2 NEW_LINE DEDENT return True NEW_LINE DEDENT
if ( isPowerOfTwo ( 31 ) ) : NEW_LINE INDENT print ( ' Yes ' ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( ' No ' ) NEW_LINE DEDENT if ( isPowerOfTwo ( 64 ) ) : NEW_LINE INDENT print ( ' Yes ' ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( ' No ' ) NEW_LINE DEDENT
def isPowerOfTwo ( x ) : NEW_LINE
return ( x and ( not ( x & ( x - 1 ) ) ) ) NEW_LINE
if ( isPowerOfTwo ( 31 ) ) : NEW_LINE INDENT print ( ' Yes ' ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( ' No ' ) NEW_LINE DEDENT if ( isPowerOfTwo ( 64 ) ) : NEW_LINE INDENT print ( ' Yes ' ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( ' No ' ) NEW_LINE DEDENT
def printTwoOdd ( arr , size ) : NEW_LINE
xor2 = arr [ 0 ] NEW_LINE
set_bit_no = 0 NEW_LINE n = size - 2 NEW_LINE x , y = 0 , 0 NEW_LINE
for i in range ( 1 , size ) : NEW_LINE INDENT xor2 = xor2 ^ arr [ i ] NEW_LINE DEDENT
set_bit_no = xor2 & ~ ( xor2 - 1 ) NEW_LINE
for i in range ( size ) : NEW_LINE
if ( arr [ i ] & set_bit_no ) : NEW_LINE INDENT x = x ^ arr [ i ] NEW_LINE DEDENT
else : NEW_LINE INDENT y = y ^ arr [ i ] NEW_LINE DEDENT print ( " The ▁ two ▁ ODD ▁ elements ▁ are " , x , " & " , y ) NEW_LINE
arr = [ 4 , 2 , 4 , 5 , 2 , 3 , 3 , 1 ] NEW_LINE arr_size = len ( arr ) NEW_LINE printTwoOdd ( arr , arr_size ) NEW_LINE
def findPair ( arr , n ) : NEW_LINE INDENT size = len ( arr ) NEW_LINE DEDENT
i , j = 0 , 1 NEW_LINE
while i < size and j < size : NEW_LINE INDENT if i != j and arr [ j ] - arr [ i ] == n : NEW_LINE INDENT print " Pair ▁ found ▁ ( " , arr [ i ] , " , " , arr [ j ] , " ) " NEW_LINE return True NEW_LINE DEDENT elif arr [ j ] - arr [ i ] < n : NEW_LINE INDENT j += 1 NEW_LINE DEDENT else : NEW_LINE INDENT i += 1 NEW_LINE DEDENT DEDENT print " No ▁ pair ▁ found " NEW_LINE return False NEW_LINE
arr = [ 1 , 8 , 30 , 40 , 100 ] NEW_LINE n = 60 NEW_LINE findPair ( arr , n ) NEW_LINE
import sys NEW_LINE
def MatrixChainOrder ( p , i , j ) : NEW_LINE INDENT if i == j : NEW_LINE INDENT return 0 NEW_LINE DEDENT _min = sys . maxsize NEW_LINE DEDENT
for k in range ( i , j ) : NEW_LINE INDENT count = ( MatrixChainOrder ( p , i , k ) + MatrixChainOrder ( p , k + 1 , j ) + p [ i - 1 ] * p [ k ] * p [ j ] ) NEW_LINE if count < _min : NEW_LINE INDENT _min = count NEW_LINE DEDENT DEDENT
return _min NEW_LINE
arr = [ 1 , 2 , 3 , 4 , 3 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Minimum ▁ number ▁ of ▁ multiplications ▁ is ▁ " , MatrixChainOrder ( arr , 1 , n - 1 ) ) NEW_LINE
def Perimeter ( s , n ) : NEW_LINE INDENT perimeter = 1 NEW_LINE DEDENT
perimeter = n * s NEW_LINE return perimeter NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
n = 5 NEW_LINE
s = 2.5 NEW_LINE
peri = Perimeter ( s , n ) NEW_LINE print ( " Perimeter ▁ of ▁ Regular ▁ Polygon ▁ with " , n , " sides ▁ of ▁ length " , s , " = " , peri ) NEW_LINE
import math NEW_LINE
def shortest_distance ( x1 , y1 , z1 , a , b , c , d ) : NEW_LINE INDENT d = abs ( ( a * x1 + b * y1 + c * z1 + d ) ) NEW_LINE e = ( math . sqrt ( a * a + b * b + c * c ) ) NEW_LINE print ( " Perpendicular ▁ distance ▁ is " , d / e ) NEW_LINE DEDENT
x1 = 4 NEW_LINE y1 = - 4 NEW_LINE z1 = 3 NEW_LINE a = 2 NEW_LINE b = - 2 NEW_LINE c = 5 NEW_LINE d = 8 NEW_LINE
shortest_distance ( x1 , y1 , z1 , a , b , c , d ) NEW_LINE
def averageOdd ( n ) : NEW_LINE INDENT if ( n % 2 == 0 ) : NEW_LINE INDENT print ( " Invalid ▁ Input " ) NEW_LINE return - 1 NEW_LINE DEDENT return ( n + 1 ) // 2 NEW_LINE DEDENT
n = 15 NEW_LINE print ( averageOdd ( n ) ) NEW_LINE
def TrinomialValue ( dp , n , k ) : NEW_LINE
if k < 0 : NEW_LINE INDENT k = - k NEW_LINE DEDENT
if dp [ n ] [ k ] != 0 : NEW_LINE INDENT return dp [ n ] [ k ] NEW_LINE DEDENT
if n == 0 and k == 0 : NEW_LINE INDENT return 1 NEW_LINE DEDENT
if k < - n or k > n : NEW_LINE INDENT return 0 NEW_LINE DEDENT
return ( TrinomialValue ( dp , n - 1 , k - 1 ) + TrinomialValue ( dp , n - 1 , k ) + TrinomialValue ( dp , n - 1 , k + 1 ) ) NEW_LINE
def printTrinomial ( n ) : NEW_LINE INDENT dp = [ [ 0 ] * 10 ] * 10 NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE
for j in range ( - i , 1 ) : NEW_LINE INDENT print ( TrinomialValue ( dp , i , j ) , end = " ▁ " ) NEW_LINE DEDENT
for j in range ( 1 , i + 1 ) : NEW_LINE INDENT print ( TrinomialValue ( dp , i , j ) , end = " ▁ " ) NEW_LINE DEDENT print ( " " , end = ' ' ) NEW_LINE
n = 4 NEW_LINE printTrinomial ( n ) NEW_LINE
def isPowerOf2 ( sttr ) : NEW_LINE INDENT len_str = len ( sttr ) ; NEW_LINE sttr = list ( sttr ) ; NEW_LINE DEDENT
num = 0 ; NEW_LINE
if ( len_str == 1 and sttr [ len_str - 1 ] == '1' ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT
while ( len_str != 1 or sttr [ len_str - 1 ] != '1' ) : NEW_LINE
if ( ( ord ( sttr [ len_str - 1 ] ) - ord ( '0' ) ) % 2 == 1 ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT
j = 0 ; NEW_LINE for i in range ( len_str ) : NEW_LINE INDENT num = num * 10 + ( ord ( sttr [ i ] ) - ord ( '0' ) ) ; NEW_LINE DEDENT
if ( num < 2 ) : NEW_LINE
if ( i != 0 ) : NEW_LINE INDENT sttr [ j ] = '0' ; NEW_LINE j += 1 ; NEW_LINE DEDENT
continue ; NEW_LINE sttr [ j ] = chr ( ( num // 2 ) + ord ( '0' ) ) ; NEW_LINE j += 1 ; NEW_LINE num = ( num ) - ( num // 2 ) * 2 ; NEW_LINE
len_str = j ; NEW_LINE
return 1 ; NEW_LINE
str1 = "124684622466842024680246842024662202000002" ; NEW_LINE str2 = "1" ; NEW_LINE str3 = "128" ; NEW_LINE print ( " " , isPowerOf2 ( str1 ) ,   " " , isPowerOf2 ( str2 ) ,   " " , isPowerOf2 ( str3 ) ) ; NEW_LINE
def averageEven ( n ) : NEW_LINE INDENT if ( n % 2 != 0 ) : NEW_LINE INDENT print ( " Invalid ▁ Input " ) NEW_LINE return - 1 NEW_LINE DEDENT return ( n + 2 ) // 2 NEW_LINE DEDENT
n = 16 NEW_LINE print ( averageEven ( n ) ) NEW_LINE
def fact ( n ) : NEW_LINE INDENT if ( n == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT return n * fact ( n - 1 ) NEW_LINE DEDENT
def div ( x ) : NEW_LINE INDENT ans = 0 ; NEW_LINE for i in range ( 1 , x + 1 ) : NEW_LINE INDENT if ( x % i == 0 ) : NEW_LINE INDENT ans += i NEW_LINE DEDENT DEDENT return ans NEW_LINE DEDENT
def sumFactDiv ( n ) : NEW_LINE INDENT return div ( fact ( n ) ) NEW_LINE DEDENT
n = 4 NEW_LINE print ( sumFactDiv ( n ) ) NEW_LINE
from math import * NEW_LINE
def printDivisors ( n ) : NEW_LINE INDENT i = 1 NEW_LINE while ( i * i < n ) : NEW_LINE INDENT if ( n % i == 0 ) : NEW_LINE INDENT print ( i , end = " ▁ " ) NEW_LINE DEDENT i += 1 NEW_LINE DEDENT for i in range ( int ( sqrt ( n ) ) , 0 , - 1 ) : NEW_LINE INDENT if ( n % i == 0 ) : NEW_LINE INDENT print ( n // i , end = " ▁ " ) NEW_LINE DEDENT DEDENT DEDENT
print ( " The ▁ divisors ▁ of ▁ 100 ▁ are : ▁ " ) NEW_LINE printDivisors ( 100 ) NEW_LINE
def printDivisors ( n ) : NEW_LINE INDENT i = 1 NEW_LINE while i <= n : NEW_LINE INDENT if ( n % i == 0 ) : NEW_LINE INDENT print i , NEW_LINE DEDENT i = i + 1 NEW_LINE DEDENT DEDENT
print " The ▁ divisors ▁ of ▁ 100 ▁ are : ▁ " NEW_LINE printDivisors ( 100 ) NEW_LINE
import math NEW_LINE
def printDivisors ( n ) : NEW_LINE
i = 1 NEW_LINE while i <= math . sqrt ( n ) : NEW_LINE INDENT if ( n % i == 0 ) : NEW_LINE DEDENT
if ( n / i == i ) : NEW_LINE INDENT print i , NEW_LINE DEDENT else : NEW_LINE
print i , n / i , NEW_LINE i = i + 1 NEW_LINE
print " The ▁ divisors ▁ of ▁ 100 ▁ are : ▁ " NEW_LINE printDivisors ( 100 ) NEW_LINE
rev_num = 0 NEW_LINE base_pos = 1 NEW_LINE
def reversDigits ( num ) : NEW_LINE INDENT global rev_num NEW_LINE global base_pos NEW_LINE if ( num > 0 ) : NEW_LINE INDENT reversDigits ( ( int ) ( num / 10 ) ) NEW_LINE rev_num += ( num % 10 ) * base_pos NEW_LINE base_pos *= 10 NEW_LINE DEDENT return rev_num NEW_LINE DEDENT
num = 4562 NEW_LINE print ( " Reverse ▁ of ▁ no . ▁ is ▁ " , reversDigits ( num ) ) NEW_LINE
def multiplyBySevenByEight ( n ) : NEW_LINE
' NEW_LINE INDENT return ( n - ( n >> 3 ) ) NEW_LINE DEDENT
n = 9 NEW_LINE print ( multiplyBySevenByEight ( n ) ) NEW_LINE
def multiplyBySevenByEight ( n ) : NEW_LINE
return ( ( n << 3 ) - n ) >> 3 ; NEW_LINE
n = 15 ; NEW_LINE print ( multiplyBySevenByEight ( n ) ) ; NEW_LINE
def binarySearch ( a , item , low , high ) : NEW_LINE INDENT while ( low <= high ) : NEW_LINE INDENT mid = low + ( high - low ) // 2 NEW_LINE if ( item == a [ mid ] ) : NEW_LINE INDENT return mid + 1 NEW_LINE DEDENT elif ( item > a [ mid ] ) : NEW_LINE INDENT low = mid + 1 NEW_LINE DEDENT else : NEW_LINE INDENT high = mid - 1 NEW_LINE DEDENT DEDENT return low NEW_LINE DEDENT
' NEW_LINE def insertionSort ( a , n ) : NEW_LINE INDENT for i in range ( n ) : NEW_LINE INDENT j = i - 1 NEW_LINE selected = a [ i ] NEW_LINE DEDENT DEDENT
loc = binarySearch ( a , selected , 0 , j ) NEW_LINE
while ( j >= loc ) : NEW_LINE INDENT a [ j + 1 ] = a [ j ] NEW_LINE j -= 1 NEW_LINE DEDENT a [ j + 1 ] = selected NEW_LINE
a = [ 37 , 23 , 0 , 17 , 12 , 72 , 31 , 46 , 100 , 88 , 54 ] NEW_LINE n = len ( a ) NEW_LINE insertionSort ( a , n ) NEW_LINE print ( " Sorted ▁ array : ▁ " ) NEW_LINE for i in range ( n ) : NEW_LINE INDENT print ( a [ i ] , end = " ▁ " ) NEW_LINE DEDENT
def insertionSort ( arr ) : NEW_LINE INDENT for i in range ( 1 , len ( arr ) ) : NEW_LINE INDENT key = arr [ i ] NEW_LINE DEDENT DEDENT
j = i - 1 NEW_LINE while j >= 0 and key < arr [ j ] : NEW_LINE INDENT arr [ j + 1 ] = arr [ j ] NEW_LINE j -= 1 NEW_LINE DEDENT arr [ j + 1 ] = key NEW_LINE
arr = [ 12 , 11 , 13 , 5 , 6 ] NEW_LINE insertionSort ( arr ) NEW_LINE for i in range ( len ( arr ) ) : NEW_LINE INDENT print ( " % ▁ d " % arr [ i ] ) NEW_LINE DEDENT
def count ( S , m , n ) : NEW_LINE
if ( n == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT
if ( n < 0 ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT
if ( m <= 0 and n >= 1 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
return count ( S , m - 1 , n ) + count ( S , m , n - S [ m - 1 ] ) ; NEW_LINE
arr = [ 1 , 2 , 3 ] NEW_LINE m = len ( arr ) NEW_LINE print ( count ( arr , m , 4 ) ) NEW_LINE
def Area ( b1 , b2 , h ) : NEW_LINE INDENT return ( ( b1 + b2 ) / 2 ) * h NEW_LINE DEDENT
base1 = 8 ; base2 = 10 ; height = 6 NEW_LINE area = Area ( base1 , base2 , height ) NEW_LINE print ( " Area ▁ is : " , area ) NEW_LINE
