def minSum ( A , N ) : NEW_LINE
mp = { } NEW_LINE sum = 0 NEW_LINE
for i in range ( N ) : NEW_LINE
sum += A [ i ] NEW_LINE
if A [ i ] in mp : NEW_LINE INDENT mp [ A [ i ] ] += 1 NEW_LINE DEDENT else : NEW_LINE INDENT mp [ A [ i ] ] = 1 NEW_LINE DEDENT
minSum = float ( ' inf ' ) NEW_LINE
for it in mp : NEW_LINE
minSum = min ( minSum , sum - ( it * mp [ it ] ) ) NEW_LINE
return minSum NEW_LINE
arr = [ 4 , 5 , 6 , 6 ] NEW_LINE
N = len ( arr ) NEW_LINE print ( minSum ( arr , N ) ) NEW_LINE
def maxAdjacent ( arr , N ) : NEW_LINE INDENT res = [ ] NEW_LINE DEDENT
for i in range ( 1 , N - 1 ) : NEW_LINE INDENT prev = arr [ 0 ] NEW_LINE DEDENT
maxi = - 1 * float ( ' inf ' ) NEW_LINE
for j in range ( 1 , N ) : NEW_LINE
if ( i == j ) : NEW_LINE INDENT continue NEW_LINE DEDENT
maxi = max ( maxi , abs ( arr [ j ] - prev ) ) NEW_LINE
prev = arr [ j ] NEW_LINE
res . append ( maxi ) NEW_LINE
for x in res : NEW_LINE INDENT print ( x , end = ' ▁ ' ) NEW_LINE DEDENT print ( ) NEW_LINE
arr = [ 1 , 3 , 4 , 7 , 8 ] NEW_LINE N = len ( arr ) NEW_LINE maxAdjacent ( arr , N ) NEW_LINE
def findSize ( N ) : NEW_LINE
if ( N == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT if ( N == 1 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT Size = 2 * findSize ( N // 2 ) + 1 NEW_LINE
return Size NEW_LINE
def CountOnes ( N , L , R ) : NEW_LINE INDENT if ( L > R ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT DEDENT
if ( N <= 1 ) : NEW_LINE INDENT return N NEW_LINE DEDENT ret = 0 NEW_LINE M = N // 2 NEW_LINE Siz_M = findSize ( M ) NEW_LINE
if ( L <= Siz_M ) : NEW_LINE
ret += CountOnes ( N // 2 , L , min ( Siz_M , R ) ) NEW_LINE
if ( L <= Siz_M + 1 and Siz_M + 1 <= R ) : NEW_LINE INDENT ret += N % 2 NEW_LINE DEDENT
if ( Siz_M + 1 < R ) : NEW_LINE INDENT ret += CountOnes ( N // 2 , max ( 1 , L - Siz_M - 1 ) , R - Siz_M - 1 ) NEW_LINE DEDENT return ret NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
N = 7 NEW_LINE L = 2 NEW_LINE R = 5 NEW_LINE
print ( CountOnes ( N , L , R ) ) NEW_LINE
def prime ( n ) : NEW_LINE
if ( n == 1 ) : NEW_LINE INDENT return False NEW_LINE DEDENT
for i in range ( 2 , n + 1 ) : NEW_LINE INDENT if i * i > n : NEW_LINE INDENT break NEW_LINE DEDENT if ( n % i == 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT
return True NEW_LINE
def minDivisior ( n ) : NEW_LINE
if ( prime ( n ) ) : NEW_LINE INDENT print ( 1 , n - 1 ) NEW_LINE DEDENT
else : NEW_LINE INDENT for i in range ( 2 , n + 1 ) : NEW_LINE INDENT if i * i > n : NEW_LINE INDENT break NEW_LINE DEDENT DEDENT DEDENT
if ( n % i == 0 ) : NEW_LINE
print ( n // i , n // i * ( i - 1 ) ) NEW_LINE break NEW_LINE
N = 4 NEW_LINE
minDivisior ( N ) NEW_LINE
import sys NEW_LINE
Landau = - sys . maxsize - 1 NEW_LINE
def gcd ( a , b ) : NEW_LINE INDENT if ( a == 0 ) : NEW_LINE INDENT return b NEW_LINE DEDENT return gcd ( b % a , a ) NEW_LINE DEDENT
def lcm ( a , b ) : NEW_LINE INDENT return ( a * b ) // gcd ( a , b ) NEW_LINE DEDENT
def findLCM ( arr ) : NEW_LINE INDENT global Landau NEW_LINE nth_lcm = arr [ 0 ] NEW_LINE for i in range ( 1 , len ( arr ) ) : NEW_LINE INDENT nth_lcm = lcm ( nth_lcm , arr [ i ] ) NEW_LINE DEDENT DEDENT
Landau = max ( Landau , nth_lcm ) NEW_LINE
def findWays ( arr , i , n ) : NEW_LINE
if ( n == 0 ) : NEW_LINE INDENT findLCM ( arr ) NEW_LINE DEDENT
for j in range ( i , n + 1 ) : NEW_LINE
arr . append ( j ) NEW_LINE
findWays ( arr , j , n - j ) NEW_LINE
arr . pop ( ) NEW_LINE
def Landau_function ( n ) : NEW_LINE INDENT arr = [ ] NEW_LINE DEDENT
findWays ( arr , 1 , n ) NEW_LINE
print ( Landau ) NEW_LINE
N = 4 NEW_LINE
Landau_function ( N ) NEW_LINE
def isPrime ( n ) : NEW_LINE
if ( n == 1 ) : NEW_LINE INDENT return True NEW_LINE DEDENT if ( n <= 3 ) : NEW_LINE INDENT return True NEW_LINE DEDENT
if ( ( n % 2 == 0 ) or ( n % 3 == 0 ) ) : NEW_LINE INDENT return False NEW_LINE DEDENT
i = 5 NEW_LINE while ( i * i <= n ) : NEW_LINE INDENT if ( ( n % i == 0 ) or ( n % ( i + 2 ) == 0 ) ) : NEW_LINE INDENT return False ; NEW_LINE i += 6 NEW_LINE DEDENT DEDENT return true ; NEW_LINE
def checkExpression ( n ) : NEW_LINE INDENT if ( isPrime ( n ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 3 NEW_LINE checkExpression ( N ) NEW_LINE DEDENT
def checkArray ( n , k , arr ) : NEW_LINE
cnt = 0 NEW_LINE for i in range ( n ) : NEW_LINE
if ( arr [ i ] & 1 ) : NEW_LINE INDENT cnt += 1 NEW_LINE DEDENT
if ( cnt >= k and cnt % 2 == k % 2 ) : NEW_LINE INDENT return True NEW_LINE DEDENT else : NEW_LINE INDENT return False NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 1 , 3 , 4 , 7 , 5 , 3 , 1 ] NEW_LINE n = len ( arr ) NEW_LINE k = 4 NEW_LINE if ( checkArray ( n , k , arr ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT DEDENT
from math import * NEW_LINE
def func ( arr , n ) : NEW_LINE INDENT ans = 0 NEW_LINE maxx = 0 NEW_LINE freq = [ 0 ] * 100005 NEW_LINE temp = 0 NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT temp = arr [ i ] NEW_LINE freq [ temp ] += 1 NEW_LINE maxx = max ( maxx , temp ) NEW_LINE DEDENT
for i in range ( 1 , maxx + 1 ) : NEW_LINE INDENT freq [ i ] += freq [ i - 1 ] NEW_LINE DEDENT for i in range ( 1 , maxx + 1 ) : NEW_LINE INDENT if ( freq [ i ] ) : NEW_LINE INDENT value = 0 NEW_LINE DEDENT DEDENT
cur = ceil ( 0.5 * i ) - 1.0 NEW_LINE j = 1.5 NEW_LINE while ( 1 ) : NEW_LINE INDENT val = min ( maxx , ( ceil ( i * j ) - 1.0 ) ) NEW_LINE times = ( freq [ i ] - freq [ i - 1 ] ) NEW_LINE con = j - 0.5 NEW_LINE DEDENT
ans += times * con * ( freq [ int ( val ) ] - freq [ int ( cur ) ] ) NEW_LINE cur = val NEW_LINE if ( val == maxx ) : NEW_LINE INDENT break NEW_LINE DEDENT j += 1 NEW_LINE
return int ( ans ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 1 , 2 , 3 ] NEW_LINE n = len ( arr ) NEW_LINE print ( func ( arr , n ) ) NEW_LINE DEDENT
def insert_element ( a , n ) : NEW_LINE
Xor = 0 NEW_LINE
Sum = 0 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT Xor ^= a [ i ] NEW_LINE Sum += a [ i ] NEW_LINE DEDENT
if ( Sum == 2 * Xor ) : NEW_LINE
print ( 0 ) NEW_LINE return NEW_LINE
if ( Xor == 0 ) : NEW_LINE INDENT print ( 1 ) NEW_LINE print ( Sum ) NEW_LINE return NEW_LINE DEDENT
num1 = Sum + Xor NEW_LINE num2 = Xor NEW_LINE
print ( 2 ) NEW_LINE
print ( num1 , num2 ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = [ 1 , 2 , 3 ] NEW_LINE n = len ( a ) NEW_LINE insert_element ( a , n ) NEW_LINE DEDENT
def checkSolution ( a , b , c ) : NEW_LINE INDENT if ( a == c ) : NEW_LINE INDENT print ( " Yes " ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) ; NEW_LINE DEDENT DEDENT
a = 2 ; b = 0 ; c = 2 ; NEW_LINE checkSolution ( a , b , c ) ; NEW_LINE
from math import * NEW_LINE
def isPerfectSquare ( x ) : NEW_LINE
sr = sqrt ( x ) NEW_LINE
return ( ( sr - floor ( sr ) ) == 0 ) NEW_LINE
def checkSunnyNumber ( N ) : NEW_LINE
if ( isPerfectSquare ( N + 1 ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT
else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE
N = 8 NEW_LINE
checkSunnyNumber ( N ) NEW_LINE
def countValues ( n ) : NEW_LINE INDENT answer = 0 NEW_LINE DEDENT
for i in range ( 2 , n + 1 , 1 ) : NEW_LINE INDENT k = n NEW_LINE DEDENT
while ( k >= i ) : NEW_LINE INDENT if ( k % i == 0 ) : NEW_LINE INDENT k //= i NEW_LINE DEDENT else : NEW_LINE INDENT k -= i NEW_LINE DEDENT DEDENT
if ( k == 1 ) : NEW_LINE INDENT answer += 1 NEW_LINE DEDENT return answer NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 6 NEW_LINE print ( countValues ( N ) ) NEW_LINE DEDENT
def printKNumbers ( N , K ) : NEW_LINE
for i in range ( K - 1 ) : NEW_LINE INDENT print ( 1 , end = ' ▁ ' ) NEW_LINE DEDENT
print ( N - K + 1 ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT ( N , K ) = ( 10 , 3 ) NEW_LINE printKNumbers ( N , K ) NEW_LINE DEDENT
def NthSmallest ( K ) : NEW_LINE
Q = [ ] NEW_LINE
for i in range ( 1 , 10 ) : NEW_LINE INDENT Q . append ( i ) NEW_LINE DEDENT
for i in range ( 1 , K + 1 ) : NEW_LINE
x = Q [ 0 ] NEW_LINE
Q . remove ( Q [ 0 ] ) NEW_LINE
if ( x % 10 != 0 ) : NEW_LINE
Q . append ( x * 10 + x % 10 - 1 ) NEW_LINE
Q . append ( x * 10 + x % 10 ) NEW_LINE
if ( x % 10 != 9 ) : NEW_LINE
Q . append ( x * 10 + x % 10 + 1 ) NEW_LINE
return x NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
N = 16 NEW_LINE print ( NthSmallest ( N ) ) NEW_LINE
from math import sqrt NEW_LINE
def nearest ( n ) : NEW_LINE
prevSquare = int ( sqrt ( n ) ) ; NEW_LINE nextSquare = prevSquare + 1 ; NEW_LINE prevSquare = prevSquare * prevSquare ; NEW_LINE nextSquare = nextSquare * nextSquare ; NEW_LINE
ans = ( prevSquare - n ) if ( n - prevSquare ) < ( nextSquare - n ) else ( nextSquare - n ) ; NEW_LINE
return ans ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 14 ; NEW_LINE print ( nearest ( n ) ) ; NEW_LINE n = 16 ; NEW_LINE print ( nearest ( n ) ) ; NEW_LINE n = 18 ; NEW_LINE print ( nearest ( n ) ) ; NEW_LINE DEDENT
from math import acos NEW_LINE
def printValueOfPi ( N ) : NEW_LINE
b = ' { : . ' + str ( N ) + ' f } ' NEW_LINE pi = b . format ( 2 * acos ( 0.0 ) ) NEW_LINE
print ( pi ) ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT N = 43 ; NEW_LINE DEDENT
printValueOfPi ( N ) ; NEW_LINE
import math NEW_LINE
def decBinary ( arr , n ) : NEW_LINE INDENT k = int ( math . log2 ( n ) ) NEW_LINE while ( n > 0 ) : NEW_LINE INDENT arr [ k ] = n % 2 NEW_LINE k = k - 1 NEW_LINE n = n // 2 NEW_LINE DEDENT DEDENT
def binaryDec ( arr , n ) : NEW_LINE INDENT ans = 0 NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT ans = ans + ( arr [ i ] << ( n - i - 1 ) ) NEW_LINE DEDENT return ans NEW_LINE DEDENT
def getNum ( n , k ) : NEW_LINE
l = int ( math . log2 ( n ) ) + 1 NEW_LINE
a = [ 0 for i in range ( 0 , l ) ] NEW_LINE decBinary ( a , n ) NEW_LINE
if ( k > l ) : NEW_LINE INDENT return n NEW_LINE DEDENT
if ( a [ k - 1 ] == 0 ) : NEW_LINE INDENT a [ k - 1 ] = 1 NEW_LINE DEDENT else : NEW_LINE INDENT a [ k - 1 ] = 0 NEW_LINE DEDENT
return binaryDec ( a , l ) NEW_LINE
n = 56 NEW_LINE k = 2 NEW_LINE print ( getNum ( n , k ) ) NEW_LINE
MAX = 1000000 NEW_LINE MOD = 10 ** 9 + 7 NEW_LINE
result = [ 0 for i in range ( MAX + 1 ) ] NEW_LINE fact = [ 0 for i in range ( MAX + 1 ) ] NEW_LINE
def preCompute ( ) : NEW_LINE
fact [ 0 ] = 1 NEW_LINE result [ 0 ] = 1 NEW_LINE
for i in range ( 1 , MAX + 1 ) : NEW_LINE
fact [ i ] = ( ( fact [ i - 1 ] % MOD ) * i ) % MOD NEW_LINE
result [ i ] = ( ( result [ i - 1 ] % MOD ) * ( fact [ i ] % MOD ) ) % MOD NEW_LINE
def performQueries ( q , n ) : NEW_LINE
preCompute ( ) NEW_LINE
for i in range ( n ) : NEW_LINE INDENT print ( result [ q [ i ] ] ) NEW_LINE DEDENT
q = [ 4 , 5 ] NEW_LINE n = len ( q ) NEW_LINE performQueries ( q , n ) NEW_LINE
import sys NEW_LINE
def gcd ( a , b ) : NEW_LINE INDENT if ( a == 0 ) : NEW_LINE INDENT return b ; NEW_LINE DEDENT return gcd ( b % a , a ) ; NEW_LINE DEDENT
def divTermCount ( a , b , c , num ) : NEW_LINE
return ( ( num / a ) + ( num / b ) + ( num / c ) - ( num / ( ( a * b ) / gcd ( a , b ) ) ) - ( num / ( ( c * b ) / gcd ( c , b ) ) ) - ( num / ( ( a * c ) / gcd ( a , c ) ) ) + ( num / ( ( a * b * c ) / gcd ( gcd ( a , b ) , c ) ) ) ) ; NEW_LINE
def findNthTerm ( a , b , c , n ) : NEW_LINE
low = 1 ; high = sys . maxsize ; mid = 0 ; NEW_LINE while ( low < high ) : NEW_LINE INDENT mid = low + ( high - low ) / 2 ; NEW_LINE DEDENT
if ( divTermCount ( a , b , c , mid ) < n ) : NEW_LINE INDENT low = mid + 1 ; NEW_LINE DEDENT
else : NEW_LINE INDENT high = mid ; NEW_LINE DEDENT return int ( low ) ; NEW_LINE
a = 2 ; b = 3 ; c = 5 ; n = 100 ; NEW_LINE print ( findNthTerm ( a , b , c , n ) ) ; NEW_LINE
def calculate_angle ( n , i , j , k ) : NEW_LINE
x , y = 0 , 0 NEW_LINE
if ( i < j ) : NEW_LINE INDENT x = j - i NEW_LINE DEDENT else : NEW_LINE INDENT x = j + n - i NEW_LINE DEDENT if ( j < k ) : NEW_LINE INDENT y = k - j NEW_LINE DEDENT else : NEW_LINE INDENT y = k + n - j NEW_LINE DEDENT
ang1 = ( 180 * x ) // n NEW_LINE ang2 = ( 180 * y ) // n NEW_LINE
ans = 180 - ang1 - ang2 NEW_LINE return ans NEW_LINE
n = 5 NEW_LINE a1 = 1 NEW_LINE a2 = 2 NEW_LINE a3 = 5 NEW_LINE print ( calculate_angle ( n , a1 , a2 , a3 ) ) NEW_LINE
def Loss ( SP , P ) : NEW_LINE INDENT loss = 0 NEW_LINE loss = ( ( 2 * P * P * SP ) / ( 100 * 100 - P * P ) ) NEW_LINE print ( " Loss ▁ = " , round ( loss , 3 ) ) NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT SP , P = 2400 , 30 NEW_LINE DEDENT
Loss ( SP , P ) NEW_LINE
MAXN = 1000001 NEW_LINE
spf = [ i for i in range ( MAXN ) ] NEW_LINE
hash1 = [ 0 for i in range ( MAXN ) ] NEW_LINE
def sieve ( ) : NEW_LINE
for i in range ( 4 , MAXN , 2 ) : NEW_LINE INDENT spf [ i ] = 2 NEW_LINE DEDENT
for i in range ( 3 , MAXN ) : NEW_LINE INDENT if i * i >= MAXN : NEW_LINE INDENT break NEW_LINE DEDENT DEDENT
if ( spf [ i ] == i ) : NEW_LINE INDENT for j in range ( i * i , MAXN , i ) : NEW_LINE DEDENT
if ( spf [ j ] == j ) : NEW_LINE INDENT spf [ j ] = i NEW_LINE DEDENT
def getFactorization ( x ) : NEW_LINE INDENT while ( x != 1 ) : NEW_LINE INDENT temp = spf [ x ] NEW_LINE if ( x % temp == 0 ) : NEW_LINE DEDENT DEDENT
hash1 [ spf [ x ] ] += 1 NEW_LINE x = x // spf [ x ] NEW_LINE while ( x % temp == 0 ) : NEW_LINE x = x // temp NEW_LINE
def check ( x ) : NEW_LINE INDENT while ( x != 1 ) : NEW_LINE INDENT temp = spf [ x ] NEW_LINE DEDENT DEDENT
if ( x % temp == 0 and hash1 [ temp ] > 1 ) : NEW_LINE INDENT return False NEW_LINE DEDENT while ( x % temp == 0 ) : NEW_LINE INDENT x = x // temp NEW_LINE DEDENT return True NEW_LINE
def hasValidNum ( arr , n ) : NEW_LINE
sieve ( ) NEW_LINE for i in range ( n ) : NEW_LINE INDENT getFactorization ( arr [ i ] ) NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT if ( check ( arr [ i ] ) ) : NEW_LINE INDENT return True NEW_LINE DEDENT DEDENT return False NEW_LINE
arr = [ 2 , 8 , 4 , 10 , 6 , 7 ] NEW_LINE n = len ( arr ) NEW_LINE if ( hasValidNum ( arr , n ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
def countWays ( N ) : NEW_LINE
E = ( N * ( N - 1 ) ) / 2 NEW_LINE if ( N == 1 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT return int ( pow ( 2 , E - 1 ) ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 4 NEW_LINE print ( countWays ( N ) ) NEW_LINE DEDENT
l = [ [ 0 for i in range ( 1001 ) ] for j in range ( 1001 ) ] NEW_LINE def initialize ( ) : NEW_LINE
l [ 0 ] [ 0 ] = 1 NEW_LINE for i in range ( 1 , 1001 ) : NEW_LINE
l [ i ] [ 0 ] = 1 NEW_LINE for j in range ( 1 , i + 1 ) : NEW_LINE
l [ i ] [ j ] = ( l [ i - 1 ] [ j - 1 ] + l [ i - 1 ] [ j ] ) NEW_LINE
def nCr ( n , r ) : NEW_LINE
return l [ n ] [ r ] NEW_LINE
initialize ( ) NEW_LINE n = 8 NEW_LINE r = 3 NEW_LINE print ( nCr ( n , r ) ) NEW_LINE
def minAbsDiff ( n ) : NEW_LINE INDENT mod = n % 4 ; NEW_LINE if ( mod == 0 or mod == 3 ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT return 1 ; NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 5 ; NEW_LINE print ( minAbsDiff ( n ) ) NEW_LINE DEDENT
def check ( s ) : NEW_LINE
freq = [ 0 ] * 10 NEW_LINE while ( s != 0 ) : NEW_LINE
r = s % 10 NEW_LINE
s = s // 10 NEW_LINE
freq [ r ] += 1 NEW_LINE xor = 0 NEW_LINE
for i in range ( 10 ) : NEW_LINE INDENT xor = xor ^ freq [ i ] NEW_LINE DEDENT if ( xor == 0 ) : NEW_LINE INDENT return True NEW_LINE DEDENT else : NEW_LINE INDENT return False NEW_LINE DEDENT
s = 122233 NEW_LINE if ( check ( s ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
def printLines ( n , k ) : NEW_LINE
for i in range ( n ) : NEW_LINE INDENT print ( k * ( 6 * i + 1 ) , k * ( 6 * i + 2 ) , k * ( 6 * i + 3 ) , k * ( 6 * i + 5 ) ) NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n , k = 2 , 2 NEW_LINE printLines ( n , k ) NEW_LINE DEDENT
def calculateSum ( n ) : NEW_LINE
return ( 2 ** ( n + 1 ) + n - 2 ) NEW_LINE
n = 4 NEW_LINE
print ( " Sum ▁ = " , calculateSum ( n ) ) NEW_LINE
mod = 1000000007 NEW_LINE
def count_special ( n ) : NEW_LINE
fib = [ 0 for i in range ( n + 1 ) ] NEW_LINE
fib [ 0 ] = 1 NEW_LINE
fib [ 1 ] = 2 NEW_LINE for i in range ( 2 , n + 1 , 1 ) : NEW_LINE
fib [ i ] = ( fib [ i - 1 ] % mod + fib [ i - 2 ] % mod ) % mod NEW_LINE
return fib [ n ] NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
n = 3 NEW_LINE print ( count_special ( n ) ) NEW_LINE
mod = 1e9 + 7 ; NEW_LINE
def ways ( i , arr , n ) : NEW_LINE
if ( i == n - 1 ) : NEW_LINE INDENT return 1 ; NEW_LINE DEDENT sum = 0 ; NEW_LINE
for j in range ( 1 , arr [ i ] + 1 ) : NEW_LINE INDENT if ( i + j < n ) : NEW_LINE INDENT sum += ( ways ( i + j , arr , n ) ) % mod ; NEW_LINE sum %= mod ; NEW_LINE DEDENT DEDENT return int ( sum % mod ) ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 5 , 3 , 1 , 4 , 3 ] ; NEW_LINE n = len ( arr ) ; NEW_LINE print ( ways ( 0 , arr , n ) ) ; NEW_LINE DEDENT
mod = 10 ** 9 + 7 NEW_LINE
def ways ( arr , n ) : NEW_LINE
dp = [ 0 ] * ( n + 1 ) NEW_LINE
dp [ n - 1 ] = 1 NEW_LINE
for i in range ( n - 2 , - 1 , - 1 ) : NEW_LINE INDENT dp [ i ] = 0 NEW_LINE DEDENT
j = 1 NEW_LINE while ( ( j + i ) < n and j <= arr [ i ] ) : NEW_LINE INDENT dp [ i ] += dp [ i + j ] NEW_LINE dp [ i ] %= mod NEW_LINE j += 1 NEW_LINE DEDENT
return dp [ 0 ] % mod NEW_LINE
arr = [ 5 , 3 , 1 , 4 , 3 ] NEW_LINE n = len ( arr ) NEW_LINE print ( ways ( arr , n ) % mod ) NEW_LINE
def countSum ( arr , n ) : NEW_LINE INDENT result = 0 NEW_LINE DEDENT
count_odd = 0 NEW_LINE count_even = 0 NEW_LINE
for i in range ( 1 , n + 1 ) : NEW_LINE
if ( arr [ i - 1 ] % 2 == 0 ) : NEW_LINE INDENT count_even = count_even + count_even + 1 NEW_LINE count_odd = count_odd + count_odd NEW_LINE DEDENT
else : NEW_LINE INDENT temp = count_even NEW_LINE count_even = count_even + count_odd NEW_LINE count_odd = count_odd + temp + 1 NEW_LINE DEDENT return ( count_even , count_odd ) NEW_LINE
arr = [ 1 , 2 , 2 , 3 ] ; NEW_LINE n = len ( arr ) NEW_LINE
count_even , count_odd = countSum ( arr , n ) ; NEW_LINE print ( " EvenSum ▁ = ▁ " , count_even , " ▁ OddSum ▁ = ▁ " , count_odd ) NEW_LINE
MAX = 10 NEW_LINE
def numToVec ( N ) : NEW_LINE INDENT digit = [ ] NEW_LINE DEDENT
while ( N != 0 ) : NEW_LINE INDENT digit . append ( N % 10 ) NEW_LINE N = N // 10 NEW_LINE DEDENT
if ( len ( digit ) == 0 ) : NEW_LINE INDENT digit . append ( 0 ) NEW_LINE DEDENT
digit = digit [ : : - 1 ] NEW_LINE
return digit NEW_LINE
def solve ( A , B , C ) : NEW_LINE INDENT d , d2 = 0 , 0 NEW_LINE DEDENT
digit = numToVec ( C ) NEW_LINE d = len ( A ) NEW_LINE
if ( B > len ( digit ) or d == 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
elif ( B < len ( digit ) ) : NEW_LINE
if ( A [ 0 ] == 0 and B != 1 ) : NEW_LINE INDENT return ( d - 1 ) * pow ( d , B - 1 ) NEW_LINE DEDENT else : NEW_LINE INDENT return pow ( d , B ) NEW_LINE DEDENT
else : NEW_LINE INDENT dp = [ 0 for i in range ( B + 1 ) ] NEW_LINE lower = [ 0 for i in range ( MAX + 1 ) ] NEW_LINE DEDENT
for i in range ( d ) : NEW_LINE INDENT lower [ A [ i ] + 1 ] = 1 NEW_LINE DEDENT for i in range ( 1 , MAX + 1 ) : NEW_LINE INDENT lower [ i ] = lower [ i - 1 ] + lower [ i ] NEW_LINE DEDENT flag = True NEW_LINE dp [ 0 ] = 0 NEW_LINE for i in range ( 1 , B + 1 ) : NEW_LINE INDENT d2 = lower [ digit [ i - 1 ] ] NEW_LINE dp [ i ] = dp [ i - 1 ] * d NEW_LINE DEDENT
if ( i == 1 and A [ 0 ] == 0 and B != 1 ) : NEW_LINE INDENT d2 = d2 - 1 NEW_LINE DEDENT
if ( flag ) : NEW_LINE INDENT dp [ i ] += d2 NEW_LINE DEDENT
flag = ( flag & ( lower [ digit [ i - 1 ] + 1 ] == lower [ digit [ i - 1 ] ] + 1 ) ) NEW_LINE return dp [ B ] NEW_LINE
A = [ 0 , 1 , 2 , 5 ] NEW_LINE N = 2 NEW_LINE k = 21 NEW_LINE print ( solve ( A , N , k ) ) NEW_LINE
import numpy as np NEW_LINE
def solve ( dp , wt , K , M , used ) : NEW_LINE
if ( wt < 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT if ( wt == 0 ) : NEW_LINE
if ( used ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT return 0 NEW_LINE if ( dp [ wt ] [ used ] != - 1 ) : NEW_LINE return dp [ wt ] [ used ] NEW_LINE ans = 0 NEW_LINE for i in range ( 1 , K + 1 ) : NEW_LINE
if ( i >= M ) : NEW_LINE INDENT ans += solve ( dp , wt - i , K , M , used 1 ) NEW_LINE DEDENT else : NEW_LINE INDENT ans += solve ( dp , wt - i , K , M , used ) NEW_LINE DEDENT dp [ wt ] [ used ] = ans NEW_LINE return ans NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT W = 3 NEW_LINE K = 3 NEW_LINE M = 2 NEW_LINE dp = np . ones ( ( W + 1 , 2 ) ) ; NEW_LINE dp = - 1 * dp NEW_LINE print ( solve ( dp , W , K , M , 0 ) ) NEW_LINE DEDENT
def partitions ( n ) : NEW_LINE INDENT p = [ 0 ] * ( n + 1 ) NEW_LINE DEDENT
p [ 0 ] = 1 NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE INDENT k = 1 NEW_LINE while ( ( k * ( 3 * k - 1 ) ) / 2 <= i ) : NEW_LINE INDENT p [ i ] += ( ( 1 if k % 2 else - 1 ) * p [ i - ( k * ( 3 * k - 1 ) ) // 2 ] ) NEW_LINE if ( k > 0 ) : NEW_LINE INDENT k *= - 1 NEW_LINE DEDENT else : NEW_LINE INDENT k = 1 - k NEW_LINE DEDENT DEDENT DEDENT return p [ n ] NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT N = 20 NEW_LINE print ( partitions ( N ) ) NEW_LINE DEDENT
MAX = 20 NEW_LINE
def LIP ( dp , mat , n , m , x , y ) : NEW_LINE
if ( dp [ x ] [ y ] < 0 ) : NEW_LINE INDENT result = 0 NEW_LINE DEDENT
if ( x == n - 1 and y == m - 1 ) : NEW_LINE INDENT dp [ x ] [ y ] = 1 NEW_LINE return dp [ x ] [ y ] NEW_LINE DEDENT
if ( x == n - 1 or y == m - 1 ) : NEW_LINE INDENT result = 1 NEW_LINE DEDENT
if ( x + 1 < n and mat [ x ] [ y ] < mat [ x + 1 ] [ y ] ) : NEW_LINE INDENT result = 1 + LIP ( dp , mat , n , m , x + 1 , y ) NEW_LINE DEDENT
if ( y + 1 < m and mat [ x ] [ y ] < mat [ x ] [ y + 1 ] ) : NEW_LINE INDENT result = max ( result , 1 + LIP ( dp , mat , n , m , x , y + 1 ) ) NEW_LINE DEDENT dp [ x ] [ y ] = result NEW_LINE return dp [ x ] [ y ] NEW_LINE
def wrapper ( mat , n , m ) : NEW_LINE INDENT dp = [ [ - 1 for i in range ( MAX ) ] for i in range ( MAX ) ] NEW_LINE return LIP ( dp , mat , n , m , 0 , 0 ) NEW_LINE DEDENT
mat = [ [ 1 , 2 , 3 , 4 ] , [ 2 , 2 , 3 , 4 ] , [ 3 , 2 , 3 , 4 ] , [ 4 , 5 , 6 , 7 ] ] NEW_LINE n = 4 NEW_LINE m = 4 NEW_LINE print ( wrapper ( mat , n , m ) ) NEW_LINE
def countPaths ( n , m ) : NEW_LINE
if ( n == 0 or m == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT
return ( countPaths ( n - 1 , m ) + countPaths ( n , m - 1 ) ) NEW_LINE
n = 3 NEW_LINE m = 2 NEW_LINE print ( " ▁ Number ▁ of ▁ Paths ▁ " , countPaths ( n , m ) ) NEW_LINE
MAX = 100 NEW_LINE
def getMaxGold ( gold , m , n ) : NEW_LINE
goldTable = [ [ 0 for i in range ( n ) ] for j in range ( m ) ] NEW_LINE for col in range ( n - 1 , - 1 , - 1 ) : NEW_LINE INDENT for row in range ( m ) : NEW_LINE DEDENT
if ( col == n - 1 ) : NEW_LINE INDENT right = 0 NEW_LINE DEDENT else : NEW_LINE INDENT right = goldTable [ row ] [ col + 1 ] NEW_LINE DEDENT
if ( row == 0 or col == n - 1 ) : NEW_LINE INDENT right_up = 0 NEW_LINE DEDENT else : NEW_LINE INDENT right_up = goldTable [ row - 1 ] [ col + 1 ] NEW_LINE DEDENT
if ( row == m - 1 or col == n - 1 ) : NEW_LINE INDENT right_down = 0 NEW_LINE DEDENT else : NEW_LINE INDENT right_down = goldTable [ row + 1 ] [ col + 1 ] NEW_LINE DEDENT
goldTable [ row ] [ col ] = gold [ row ] [ col ] + max ( right , right_up , right_down ) NEW_LINE
res = goldTable [ 0 ] [ 0 ] NEW_LINE for i in range ( 1 , m ) : NEW_LINE INDENT res = max ( res , goldTable [ i ] [ 0 ] ) NEW_LINE DEDENT return res NEW_LINE
gold = [ [ 1 , 3 , 1 , 5 ] , [ 2 , 2 , 4 , 1 ] , [ 5 , 0 , 2 , 3 ] , [ 0 , 6 , 1 , 2 ] ] NEW_LINE m = 4 NEW_LINE n = 4 NEW_LINE print ( getMaxGold ( gold , m , n ) ) NEW_LINE
M = 100 NEW_LINE
def minAdjustmentCost ( A , n , target ) : NEW_LINE
dp = [ [ 0 for i in range ( M + 1 ) ] for i in range ( n ) ] NEW_LINE
for j in range ( M + 1 ) : NEW_LINE INDENT dp [ 0 ] [ j ] = abs ( j - A [ 0 ] ) NEW_LINE DEDENT
for i in range ( 1 , n ) : NEW_LINE
for j in range ( M + 1 ) : NEW_LINE
dp [ i ] [ j ] = 100000000 NEW_LINE
for k in range ( max ( j - target , 0 ) , min ( M , j + target ) + 1 ) : NEW_LINE INDENT dp [ i ] [ j ] = min ( dp [ i ] [ j ] , dp [ i - 1 ] [ k ] + abs ( A [ i ] - j ) ) NEW_LINE DEDENT
res = 10000000 NEW_LINE for j in range ( M + 1 ) : NEW_LINE INDENT res = min ( res , dp [ n - 1 ] [ j ] ) NEW_LINE DEDENT return res NEW_LINE
arr = [ 55 , 77 , 52 , 61 , 39 , 6 , 25 , 60 , 49 , 47 ] NEW_LINE n = len ( arr ) NEW_LINE target = 10 NEW_LINE print ( " Minimum ▁ adjustment ▁ cost ▁ is " , minAdjustmentCost ( arr , n , target ) , sep = ' ▁ ' ) NEW_LINE
def totalCombination ( L , R ) : NEW_LINE
count = 0 NEW_LINE
K = R - L NEW_LINE
if ( K < L ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
ans = K - L NEW_LINE
count = ( ( ans + 1 ) * ( ans + 2 ) ) // 2 NEW_LINE
return count NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT L , R = 2 , 6 NEW_LINE print ( totalCombination ( L , R ) ) NEW_LINE DEDENT
def printArrays ( n ) : NEW_LINE
A , B = [ ] , [ ] ; NEW_LINE
for i in range ( 1 , 2 * n + 1 ) : NEW_LINE
if ( i % 2 == 0 ) : NEW_LINE INDENT A . append ( i ) ; NEW_LINE DEDENT else : NEW_LINE INDENT B . append ( i ) ; NEW_LINE DEDENT
print ( " { ▁ " , end = " " ) ; NEW_LINE for i in range ( n ) : NEW_LINE INDENT print ( A [ i ] , end = " " ) ; NEW_LINE if ( i != n - 1 ) : NEW_LINE INDENT print ( " , ▁ " , end = " " ) ; NEW_LINE DEDENT DEDENT print ( " } " ) ; NEW_LINE
print ( " { ▁ " , end = " " ) ; NEW_LINE for i in range ( n ) : NEW_LINE INDENT print ( B [ i ] , end = " " ) ; NEW_LINE if ( i != n - 1 ) : NEW_LINE INDENT print ( " , " , end = " ▁ " ) ; NEW_LINE DEDENT DEDENT print ( " ▁ } " , end = " " ) ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT N = 5 ; NEW_LINE DEDENT
printArrays ( N ) ; NEW_LINE
def flipBitsOfAandB ( A , B ) : NEW_LINE
for i in range ( 0 , 32 ) : NEW_LINE
if ( ( A & ( 1 << i ) ) and ( B & ( 1 << i ) ) ) : NEW_LINE
A = A ^ ( 1 << i ) NEW_LINE
B = B ^ ( 1 << i ) NEW_LINE
print ( A , B ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT A = 7 NEW_LINE B = 4 NEW_LINE flipBitsOfAandB ( A , B ) NEW_LINE DEDENT
def findDistinctSums ( N ) : NEW_LINE INDENT return ( 2 * N - 1 ) NEW_LINE DEDENT
N = 3 NEW_LINE print ( findDistinctSums ( N ) ) NEW_LINE
def countSubstrings ( str ) : NEW_LINE
freq = [ 0 ] * 3 NEW_LINE
count = 0 NEW_LINE i = 0 NEW_LINE
for j in range ( 0 , len ( str ) ) : NEW_LINE
freq [ ord ( str [ j ] ) - ord ( '0' ) ] += 1 NEW_LINE
while ( freq [ 0 ] > 0 and freq [ 1 ] > 0 and freq [ 2 ] > 0 ) : NEW_LINE INDENT i += 1 NEW_LINE freq [ ord ( str [ i ] ) - ord ( '0' ) ] -= 1 NEW_LINE DEDENT
count += i NEW_LINE
return count NEW_LINE
str = "00021" NEW_LINE count = countSubstrings ( str ) NEW_LINE print ( count ) NEW_LINE
def minFlips ( st ) : NEW_LINE
count = 0 NEW_LINE
if ( len ( st ) <= 2 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
for i in range ( len ( st ) - 2 ) : NEW_LINE
if ( st [ i ] == st [ i + 1 ] and st [ i + 2 ] == st [ i + 1 ] ) : NEW_LINE INDENT i = i + 3 NEW_LINE count += 1 NEW_LINE DEDENT else : NEW_LINE INDENT i += 1 NEW_LINE DEDENT
return count NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT S = "0011101" NEW_LINE print ( minFlips ( S ) ) NEW_LINE DEDENT
def convertToHex ( num ) : NEW_LINE INDENT temp = " " NEW_LINE while ( num != 0 ) : NEW_LINE INDENT rem = num % 16 NEW_LINE c = 0 NEW_LINE if ( rem < 10 ) : NEW_LINE INDENT c = rem + 48 NEW_LINE DEDENT else : NEW_LINE INDENT c = rem + 87 NEW_LINE DEDENT temp += chr ( c ) NEW_LINE num = num // 16 NEW_LINE DEDENT return temp NEW_LINE DEDENT
def encryptString ( S , N ) : NEW_LINE INDENT ans = " " NEW_LINE DEDENT
for i in range ( N ) : NEW_LINE INDENT ch = S [ i ] NEW_LINE count = 0 NEW_LINE DEDENT
while ( i < N and S [ i ] == ch ) : NEW_LINE
count += 1 NEW_LINE i += 1 NEW_LINE
i -= 1 NEW_LINE
hex = convertToHex ( count ) NEW_LINE
ans += ch NEW_LINE
ans += hex NEW_LINE
ans = ans [ : : - 1 ] NEW_LINE
return ans NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
S = " abc " NEW_LINE N = len ( S ) NEW_LINE
print ( encryptString ( S , N ) ) NEW_LINE
def binomialCoeff ( n , k ) : NEW_LINE INDENT res = 1 NEW_LINE DEDENT
if ( k > n - k ) : NEW_LINE INDENT k = n - k NEW_LINE DEDENT
for i in range ( k ) : NEW_LINE INDENT res *= ( n - i ) NEW_LINE res //= ( i + 1 ) NEW_LINE DEDENT return res NEW_LINE
def countOfString ( N ) : NEW_LINE
Stotal = pow ( 2 , N ) NEW_LINE
Sequal = 0 NEW_LINE
if ( N % 2 == 0 ) : NEW_LINE INDENT Sequal = binomialCoeff ( N , N // 2 ) NEW_LINE DEDENT S1 = ( Stotal - Sequal ) // 2 NEW_LINE return S1 NEW_LINE
N = 3 NEW_LINE print ( countOfString ( N ) ) NEW_LINE
def removeCharRecursive ( str , X ) : NEW_LINE
if ( len ( str ) == 0 ) : NEW_LINE INDENT return " " NEW_LINE DEDENT
if ( str [ 0 ] == X ) : NEW_LINE
return removeCharRecursive ( str [ 1 : ] , X ) NEW_LINE
return str [ 0 ] + removeCharRecursive ( str [ 1 : ] , X ) NEW_LINE
str = " geeksforgeeks " NEW_LINE
X = ' e ' NEW_LINE
str = removeCharRecursive ( str , X ) NEW_LINE print ( str ) NEW_LINE
def isValid ( a1 , a2 , strr , flag ) : NEW_LINE INDENT v1 , v2 = 0 , 0 NEW_LINE DEDENT
if ( flag == 0 ) : NEW_LINE INDENT v1 = strr [ 4 ] NEW_LINE v2 = strr [ 3 ] NEW_LINE DEDENT else : NEW_LINE
v1 = strr [ 1 ] NEW_LINE v2 = strr [ 0 ] NEW_LINE
if ( v1 != a1 and v1 != ' ? ' ) : NEW_LINE INDENT return False NEW_LINE DEDENT if ( v2 != a2 and v2 != ' ? ' ) : NEW_LINE INDENT return False NEW_LINE DEDENT return True NEW_LINE
def inRange ( hh , mm , L , R ) : NEW_LINE INDENT a = abs ( hh - mm ) NEW_LINE DEDENT
if ( a < L or a > R ) : NEW_LINE INDENT return False NEW_LINE DEDENT return True NEW_LINE
def displayTime ( hh , mm ) : NEW_LINE INDENT if ( hh > 10 ) : NEW_LINE INDENT print ( hh , end = " : " ) NEW_LINE DEDENT elif ( hh < 10 ) : NEW_LINE INDENT print ( "0" , hh , end = " : " ) NEW_LINE DEDENT if ( mm > 10 ) : NEW_LINE INDENT print ( mm ) NEW_LINE DEDENT elif ( mm < 10 ) : NEW_LINE INDENT print ( "0" , mm ) NEW_LINE DEDENT DEDENT
def maximumTimeWithDifferenceInRange ( strr , L , R ) : NEW_LINE INDENT i , j = 0 , 0 NEW_LINE h1 , h2 , m1 , m2 = 0 , 0 , 0 , 0 NEW_LINE DEDENT
for i in range ( 23 , - 1 , - 1 ) : NEW_LINE INDENT h1 = i % 10 NEW_LINE h2 = i // 10 NEW_LINE DEDENT
if ( not isValid ( chr ( h1 ) , chr ( h2 ) , strr , 1 ) ) : NEW_LINE INDENT continue NEW_LINE DEDENT
for j in range ( 59 , - 1 , - 1 ) : NEW_LINE INDENT m1 = j % 10 NEW_LINE m2 = j // 10 NEW_LINE DEDENT
if ( not isValid ( chr ( m1 ) , chr ( m2 ) , strr , 0 ) ) : NEW_LINE INDENT continue NEW_LINE DEDENT if ( inRange ( i , j , L , R ) ) : NEW_LINE INDENT displayTime ( i , j ) NEW_LINE return NEW_LINE DEDENT if ( inRange ( i , j , L , R ) ) : NEW_LINE displayTime ( i , j ) NEW_LINE else : NEW_LINE print ( - 1 ) NEW_LINE
timeValue = " ? ? : ? ? " NEW_LINE
L = 20 NEW_LINE R = 39 NEW_LINE maximumTimeWithDifferenceInRange ( timeValue , L , R ) NEW_LINE
def check ( s , n ) : NEW_LINE
st = [ ] NEW_LINE
for i in range ( n ) : NEW_LINE
if ( len ( st ) != 0 and st [ len ( st ) - 1 ] == s [ i ] ) : NEW_LINE INDENT st . pop ( ) ; NEW_LINE DEDENT
else : NEW_LINE INDENT st . append ( s [ i ] ) ; NEW_LINE DEDENT
if ( len ( st ) == 0 ) : NEW_LINE INDENT return True ; NEW_LINE DEDENT
else : NEW_LINE INDENT return False ; NEW_LINE DEDENT
str = " aanncddc " ; NEW_LINE n = len ( str ) NEW_LINE
if ( check ( str , n ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
from collections import defaultdict NEW_LINE def findNumOfValidWords ( w , p ) : NEW_LINE
m = defaultdict ( int ) NEW_LINE
res = [ ] NEW_LINE
for s in w : NEW_LINE INDENT val = 0 NEW_LINE DEDENT
for c in s : NEW_LINE INDENT val = val | ( 1 << ( ord ( c ) - ord ( ' a ' ) ) ) NEW_LINE DEDENT
m [ val ] += 1 NEW_LINE
for s in p : NEW_LINE INDENT val = 0 NEW_LINE DEDENT
for c in s : NEW_LINE INDENT val = val | ( 1 << ( ord ( c ) - ord ( ' a ' ) ) ) NEW_LINE DEDENT temp = val NEW_LINE first = ord ( s [ 0 ] ) - ord ( ' a ' ) NEW_LINE count = 0 NEW_LINE while ( temp != 0 ) : NEW_LINE
if ( ( ( temp >> first ) & 1 ) == 1 ) : NEW_LINE INDENT if ( temp in m ) : NEW_LINE INDENT count += m [ temp ] NEW_LINE DEDENT DEDENT
temp = ( temp - 1 ) & val NEW_LINE
res . append ( count ) NEW_LINE
for it in res : NEW_LINE INDENT print ( it ) NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr1 = [ " aaaa " , " asas " , " able " , " ability " , " actt " , " actor " , " access " ] NEW_LINE arr2 = [ " aboveyz " , " abrodyz " , " absolute " , " absoryz " , " actresz " , " gaswxyz " ] NEW_LINE DEDENT
findNumOfValidWords ( arr1 , arr2 ) NEW_LINE
def flip ( s ) : NEW_LINE INDENT s = list ( s ) NEW_LINE for i in range ( len ( s ) ) : NEW_LINE DEDENT
if ( s [ i ] == '0' ) : NEW_LINE
while ( s [ i ] == '0' ) : NEW_LINE
s [ i ] = '1' NEW_LINE i += 1 NEW_LINE s = ' ' . join ( map ( str , s ) ) NEW_LINE
return s NEW_LINE
s = "100010001" NEW_LINE print ( flip ( s ) ) NEW_LINE
def getOrgString ( s ) : NEW_LINE
print ( s [ 0 ] , end = " " ) NEW_LINE
i = 1 NEW_LINE while ( i < len ( s ) ) : NEW_LINE
if ( ord ( s [ i ] ) >= ord ( ' A ' ) and ord ( s [ i ] ) <= ord ( ' Z ' ) ) : NEW_LINE INDENT print ( " ▁ " , s [ i ] . lower ( ) , end = " " ) NEW_LINE DEDENT
else : NEW_LINE INDENT print ( s [ i ] , end = " " ) NEW_LINE DEDENT i += 1 NEW_LINE
s = " ILoveGeeksForGeeks " NEW_LINE getOrgString ( s ) NEW_LINE
' NEW_LINE def countChar ( str , x ) : NEW_LINE INDENT count = 0 NEW_LINE for i in range ( len ( str ) ) : NEW_LINE INDENT if ( str [ i ] == x ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT DEDENT n = 10 NEW_LINE DEDENT
repetitions = n // len ( str ) NEW_LINE count = count * repetitions NEW_LINE
l = n % len ( str ) NEW_LINE for i in range ( l ) : NEW_LINE INDENT if ( str [ i ] == x ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT DEDENT return count NEW_LINE
str = " abcac " NEW_LINE print ( countChar ( str , ' a ' ) ) NEW_LINE
def countFreq ( arr , n , limit ) : NEW_LINE
count = [ 0 for i in range ( limit + 1 ) ] NEW_LINE
for i in range ( n ) : NEW_LINE INDENT count [ arr [ i ] ] += 1 NEW_LINE DEDENT for i in range ( limit + 1 ) : NEW_LINE INDENT if ( count [ i ] > 0 ) : NEW_LINE INDENT print ( i , count [ i ] ) NEW_LINE DEDENT DEDENT
arr = [ 5 , 5 , 6 , 6 , 5 , 6 , 1 , 2 , 3 , 10 , 10 ] NEW_LINE n = len ( arr ) NEW_LINE limit = 10 NEW_LINE countFreq ( arr , n , limit ) NEW_LINE
def check ( s , m ) : NEW_LINE
l = len ( s ) ; NEW_LINE
c1 = 0 ; NEW_LINE
c2 = 0 ; NEW_LINE for i in range ( 0 , l - 1 ) : NEW_LINE INDENT if ( s [ i ] == '0' ) : NEW_LINE INDENT c2 = 0 ; NEW_LINE DEDENT DEDENT
c1 = c1 + 1 ; NEW_LINE else : NEW_LINE c1 = 0 ; NEW_LINE
c2 = c2 + 1 ; NEW_LINE if ( c1 == m or c2 == m ) : NEW_LINE return True ; NEW_LINE return False ; NEW_LINE
s = "001001" ; NEW_LINE m = 2 ; NEW_LINE
if ( check ( s , m ) ) : NEW_LINE INDENT print ( " YES " ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NO " ) ; NEW_LINE DEDENT
def productAtKthLevel ( tree , k ) : NEW_LINE INDENT level = - 1 NEW_LINE DEDENT
product = 1 NEW_LINE n = len ( tree ) NEW_LINE for i in range ( 0 , n ) : NEW_LINE
if ( tree [ i ] == ' ( ' ) : NEW_LINE INDENT level += 1 NEW_LINE DEDENT
elif ( tree [ i ] == ' ) ' ) : NEW_LINE INDENT level -= 1 NEW_LINE DEDENT else : NEW_LINE
if ( level == k ) : NEW_LINE INDENT product *= ( int ( tree [ i ] ) - int ( '0' ) ) NEW_LINE DEDENT
return product NEW_LINE
tree = " ( 0(5(6 ( ) ( ) ) ( 4 ( ) (9 ( ) ( ) ) ) ) ( 7(1 ( ) ( ) ) ( 3 ( ) ( ) ) ) ) " NEW_LINE k = 2 NEW_LINE print ( productAtKthLevel ( tree , k ) ) NEW_LINE
def findDuplicates ( a , n , m ) : NEW_LINE
isPresent = [ [ False for i in range ( n ) ] for j in range ( m ) ] NEW_LINE for i in range ( n ) : NEW_LINE INDENT for j in range ( m ) : NEW_LINE DEDENT
for k in range ( n ) : NEW_LINE INDENT if i != k and a [ i ] [ j ] == a [ k ] [ j ] : NEW_LINE INDENT isPresent [ i ] [ j ] = True NEW_LINE isPresent [ k ] [ j ] = True NEW_LINE DEDENT DEDENT
for k in range ( m ) : NEW_LINE INDENT if j != k and a [ i ] [ j ] == a [ i ] [ k ] : NEW_LINE INDENT isPresent [ i ] [ j ] = True NEW_LINE isPresent [ i ] [ k ] = True NEW_LINE DEDENT DEDENT for i in range ( n ) : NEW_LINE for j in range ( m ) : NEW_LINE
if not isPresent [ i ] [ j ] : NEW_LINE INDENT print ( a [ i ] [ j ] , end = " " ) NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 2 NEW_LINE m = 2 NEW_LINE DEDENT
a = [ " zx " , " xz " ] NEW_LINE
findDuplicates ( a , n , m ) NEW_LINE
def isValidISBN ( isbn ) : NEW_LINE
if len ( isbn ) != 10 : NEW_LINE INDENT return False NEW_LINE DEDENT
_sum = 0 NEW_LINE for i in range ( 9 ) : NEW_LINE INDENT if 0 <= int ( isbn [ i ] ) <= 9 : NEW_LINE INDENT _sum += int ( isbn [ i ] ) * ( 10 - i ) NEW_LINE DEDENT else : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT
if ( isbn [ 9 ] != ' X ' and 0 <= int ( isbn [ 9 ] ) <= 9 ) : NEW_LINE INDENT return False NEW_LINE DEDENT
_sum += 10 if isbn [ 9 ] == ' X ' else int ( isbn [ 9 ] ) NEW_LINE
return ( _sum % 11 == 0 ) NEW_LINE
isbn = "007462542X " NEW_LINE if isValidISBN ( isbn ) : NEW_LINE INDENT print ( ' Valid ' ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Invalid " ) NEW_LINE DEDENT
def isVowel ( c ) : NEW_LINE INDENT if ( c == ' a ' or c == ' A ' or c == ' e ' or c == ' E ' or c == ' i ' or c == ' I ' or c == ' o ' or c == ' O ' or c == ' u ' or c == ' U ' ) : NEW_LINE INDENT return True NEW_LINE DEDENT return False NEW_LINE DEDENT
def reverserVowel ( string ) : NEW_LINE INDENT j = 0 NEW_LINE vowel = [ 0 ] * len ( string ) NEW_LINE string = list ( string ) NEW_LINE DEDENT
for i in range ( len ( string ) ) : NEW_LINE INDENT if isVowel ( string [ i ] ) : NEW_LINE INDENT vowel [ j ] = string [ i ] NEW_LINE j += 1 NEW_LINE DEDENT DEDENT
for i in range ( len ( string ) ) : NEW_LINE INDENT if isVowel ( string [ i ] ) : NEW_LINE INDENT j -= 1 NEW_LINE string [ i ] = vowel [ j ] NEW_LINE DEDENT DEDENT return ' ' . join ( string ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT string = " hello ▁ world " NEW_LINE print ( reverserVowel ( string ) ) NEW_LINE DEDENT
def firstLetterWord ( str ) : NEW_LINE INDENT result = " " NEW_LINE DEDENT
v = True NEW_LINE for i in range ( len ( str ) ) : NEW_LINE
if ( str [ i ] == ' ▁ ' ) : NEW_LINE INDENT v = True NEW_LINE DEDENT
elif ( str [ i ] != ' ▁ ' and v == True ) : NEW_LINE INDENT result += ( str [ i ] ) NEW_LINE v = False NEW_LINE DEDENT return result NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT str = " geeks ▁ for ▁ geeks " NEW_LINE print ( firstLetterWord ( str ) ) NEW_LINE DEDENT
def dfs ( i , j , grid , vis , ans , z , z_count ) : NEW_LINE INDENT n = len ( grid ) NEW_LINE m = len ( grid [ 0 ] ) NEW_LINE DEDENT
vis [ i ] [ j ] = 1 NEW_LINE if ( grid [ i ] [ j ] == 0 ) : NEW_LINE
z += 1 NEW_LINE
if ( grid [ i ] [ j ] == 2 ) : NEW_LINE
if ( z == z_count ) : NEW_LINE INDENT ans += 1 NEW_LINE DEDENT vis [ i ] [ j ] = 0 NEW_LINE return grid , vis , ans NEW_LINE
if ( i >= 1 and not vis [ i - 1 ] [ j ] and grid [ i - 1 ] [ j ] != - 1 ) : NEW_LINE INDENT grid , vis , ans = dfs ( i - 1 , j , grid , vis , ans , z , z_count ) NEW_LINE DEDENT
if ( i < n - 1 and not vis [ i + 1 ] [ j ] and grid [ i + 1 ] [ j ] != - 1 ) : NEW_LINE INDENT grid , vis , ans = dfs ( i + 1 , j , grid , vis , ans , z , z_count ) NEW_LINE DEDENT
if ( j >= 1 and not vis [ i ] [ j - 1 ] and grid [ i ] [ j - 1 ] != - 1 ) : NEW_LINE INDENT grid , vis , ans = dfs ( i , j - 1 , grid , vis , ans , z , z_count ) NEW_LINE DEDENT
if ( j < m - 1 and not vis [ i ] [ j + 1 ] and grid [ i ] [ j + 1 ] != - 1 ) : NEW_LINE INDENT grid , vis , ans = dfs ( i , j + 1 , grid , vis , ans , z , z_count ) NEW_LINE DEDENT
vis [ i ] [ j ] = 0 NEW_LINE return grid , vis , ans NEW_LINE
def uniquePaths ( grid ) : NEW_LINE
z_count = 0 NEW_LINE n = len ( grid ) NEW_LINE m = len ( grid [ 0 ] ) NEW_LINE ans = 0 NEW_LINE vis = [ [ 0 for j in range ( m ) ] for i in range ( n ) ] NEW_LINE x = 0 NEW_LINE y = 0 NEW_LINE for i in range ( n ) : NEW_LINE INDENT for j in range ( m ) : NEW_LINE DEDENT
if grid [ i ] [ j ] == 0 : NEW_LINE INDENT z_count += 1 NEW_LINE DEDENT elif ( grid [ i ] [ j ] == 1 ) : NEW_LINE
x = i NEW_LINE y = j NEW_LINE grid , vis , ans = dfs ( x , y , grid , vis , ans , 0 , z_count ) NEW_LINE return ans NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT grid = [ [ 1 , 0 , 0 , 0 ] , [ 0 , 0 , 0 , 0 ] , [ 0 , 0 , 2 , - 1 ] ] NEW_LINE print ( uniquePaths ( grid ) ) NEW_LINE DEDENT
def numPairs ( a , n ) : NEW_LINE
ans = 0 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT a [ i ] = abs ( a [ i ] ) NEW_LINE DEDENT
a . sort ( ) NEW_LINE
for i in range ( n ) : NEW_LINE INDENT index = 0 NEW_LINE for j in range ( i + 1 , n ) : NEW_LINE INDENT if ( 2 * a [ i ] >= a [ j - 1 ] and 2 * a [ i ] < a [ j ] ) : NEW_LINE INDENT index = j NEW_LINE DEDENT DEDENT if index == 0 : NEW_LINE INDENT index = n NEW_LINE DEDENT ans += index - i - 1 NEW_LINE DEDENT
return ans NEW_LINE
a = [ 3 , 6 ] NEW_LINE n = len ( a ) NEW_LINE print ( numPairs ( a , n ) ) NEW_LINE
def areaOfSquare ( S ) : NEW_LINE
area = S * S NEW_LINE return area NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
S = 5 NEW_LINE
print ( areaOfSquare ( S ) ) NEW_LINE
def maxPointOfIntersection ( x , y ) : NEW_LINE INDENT k = y * ( y - 1 ) // 2 NEW_LINE k = k + x * ( 2 * y + x - 1 ) NEW_LINE return k NEW_LINE DEDENT
x = 3 NEW_LINE
y = 4 NEW_LINE
print ( maxPointOfIntersection ( x , y ) ) NEW_LINE
def Icosihenagonal_num ( n ) : NEW_LINE
return ( 19 * n * n - 17 * n ) / 2 NEW_LINE
n = 3 NEW_LINE print ( int ( Icosihenagonal_num ( n ) ) ) NEW_LINE n = 10 NEW_LINE print ( int ( Icosihenagonal_num ( n ) ) ) NEW_LINE
def find_Centroid ( v ) : NEW_LINE INDENT ans = [ 0 , 0 ] NEW_LINE n = len ( v ) NEW_LINE signedArea = 0 NEW_LINE DEDENT
for i in range ( len ( v ) ) : NEW_LINE INDENT x0 = v [ i ] [ 0 ] NEW_LINE y0 = v [ i ] [ 1 ] NEW_LINE x1 = v [ ( i + 1 ) % n ] [ 0 ] NEW_LINE y1 = v [ ( i + 1 ) % n ] [ 1 ] NEW_LINE DEDENT
A = ( x0 * y1 ) - ( x1 * y0 ) NEW_LINE signedArea += A NEW_LINE
ans [ 0 ] += ( x0 + x1 ) * A NEW_LINE ans [ 1 ] += ( y0 + y1 ) * A NEW_LINE signedArea *= 0.5 NEW_LINE ans [ 0 ] = ( ans [ 0 ] ) / ( 6 * signedArea ) NEW_LINE ans [ 1 ] = ( ans [ 1 ] ) / ( 6 * signedArea ) NEW_LINE return ans NEW_LINE
vp = [ [ 1 , 2 ] , [ 3 , - 4 ] , [ 6 , - 7 ] ] NEW_LINE ans = find_Centroid ( vp ) NEW_LINE print ( round ( ans [ 0 ] , 12 ) , ans [ 1 ] ) NEW_LINE
d = 10 NEW_LINE a = 0.0 NEW_LINE
a = ( 360 - ( 6 * d ) ) / 4 NEW_LINE
print ( a , " , " , a + d , " , " , a + 2 * d , " , " , a + 3 * d , sep = ' ▁ ' ) NEW_LINE
import math NEW_LINE
def distance ( a1 , b1 , c1 , d1 , a2 , b2 , c2 , d2 ) : NEW_LINE INDENT if ( a1 / a2 == b1 / b2 and b1 / b2 == c1 / c2 ) : NEW_LINE INDENT x1 = y1 = 0 NEW_LINE z1 = - d1 / c1 NEW_LINE d = abs ( ( c2 * z1 + d2 ) ) / ( math . sqrt ( a2 * a2 + b2 * b2 + c2 * c2 ) ) NEW_LINE print ( " Perpendicular ▁ distance ▁ is " ) , d NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Planes ▁ are ▁ not ▁ parallel " ) NEW_LINE DEDENT DEDENT
a1 = 1 NEW_LINE b1 = 2 NEW_LINE c1 = - 1 NEW_LINE d1 = 1 NEW_LINE a2 = 3 NEW_LINE b2 = 6 NEW_LINE c2 = - 3 NEW_LINE d2 = - 4 NEW_LINE distance ( a1 , b1 , c1 , d1 , a2 , b2 , c2 , d2 ) NEW_LINE
def factorial ( n ) : NEW_LINE INDENT if ( n == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT return n * factorial ( n - 1 ) NEW_LINE DEDENT
def numOfNecklace ( N ) : NEW_LINE
ans = factorial ( N ) // ( factorial ( N // 2 ) * factorial ( N // 2 ) ) NEW_LINE
ans = ans * factorial ( N // 2 - 1 ) NEW_LINE ans = ans * factorial ( N // 2 - 1 ) NEW_LINE
ans //= 2 NEW_LINE
return ans NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
N = 4 NEW_LINE
print ( numOfNecklace ( N ) ) NEW_LINE
def isDivisibleByDivisor ( S , D ) : NEW_LINE
S %= D NEW_LINE
hashMap = set ( ) NEW_LINE hashMap . add ( S ) NEW_LINE for i in range ( D + 1 ) : NEW_LINE
S += ( S % D ) NEW_LINE S %= D NEW_LINE
if ( S in hashMap ) : NEW_LINE
if ( S == 0 ) : NEW_LINE INDENT return " Yes " NEW_LINE DEDENT return " No " NEW_LINE
else : NEW_LINE INDENT hashMap . add ( S ) NEW_LINE DEDENT return " Yes " NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT S = 3 NEW_LINE D = 6 NEW_LINE print ( isDivisibleByDivisor ( S , D ) ) NEW_LINE DEDENT
def minimumSteps ( x , y ) : NEW_LINE
cnt = 0 NEW_LINE
while ( x != 0 and y != 0 ) : NEW_LINE
if ( x > y ) : NEW_LINE
cnt += x / y NEW_LINE x %= y NEW_LINE
else : NEW_LINE
cnt += y / x NEW_LINE y %= x NEW_LINE cnt -= 1 NEW_LINE
if ( x > 1 or y > 1 ) : NEW_LINE INDENT cnt = - 1 NEW_LINE DEDENT
print ( int ( cnt ) ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
x = 3 NEW_LINE y = 1 NEW_LINE minimumSteps ( x , y ) NEW_LINE
def countMinReversals ( expr ) : NEW_LINE INDENT lenn = len ( expr ) NEW_LINE DEDENT
if ( lenn % 2 ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
s = [ ] NEW_LINE for i in range ( lenn ) : NEW_LINE INDENT if ( expr [ i ] == ' ' and len ( s ) ) : NEW_LINE INDENT if ( s [ 0 ] == ' ' ) : NEW_LINE INDENT s . pop ( 0 ) NEW_LINE DEDENT else : NEW_LINE INDENT s . insert ( 0 , expr [ i ] ) NEW_LINE DEDENT DEDENT else : NEW_LINE INDENT s . insert ( 0 , expr [ i ] ) NEW_LINE DEDENT DEDENT
red_len = len ( s ) NEW_LINE
n = 0 NEW_LINE while ( len ( s ) and s [ 0 ] == ' ' ) : NEW_LINE INDENT s . pop ( 0 ) NEW_LINE n += 1 NEW_LINE DEDENT
return ( red_len // 2 + n % 2 ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT expr = " } } { { " NEW_LINE print ( countMinReversals ( expr . strip ( ) ) ) NEW_LINE DEDENT
def PrintMinNumberForPattern ( arr ) : NEW_LINE
curr_max = 0 NEW_LINE
last_entry = 0 NEW_LINE i = 0 NEW_LINE
while i < len ( arr ) : NEW_LINE
noOfNextD = 0 NEW_LINE if arr [ i ] == " I " : NEW_LINE
j = i + 1 NEW_LINE while j < len ( arr ) and arr [ j ] == " D " : NEW_LINE INDENT noOfNextD += 1 NEW_LINE j += 1 NEW_LINE DEDENT if i == 0 : NEW_LINE INDENT curr_max = noOfNextD + 2 NEW_LINE last_entry += 1 NEW_LINE DEDENT
print ( " " , last_entry , end = " " ) NEW_LINE print ( " " , curr_max , end = " " ) NEW_LINE
last_entry = curr_max NEW_LINE else : NEW_LINE
curr_max += noOfNextD + 1 NEW_LINE
last_entry = curr_max NEW_LINE print ( " " , last_entry , end = " " ) NEW_LINE
for k in range ( noOfNextD ) : NEW_LINE INDENT last_entry -= 1 NEW_LINE print ( " " , last_entry , end = " " ) NEW_LINE i += 1 NEW_LINE DEDENT
elif arr [ i ] == " D " : NEW_LINE INDENT if i == 0 : NEW_LINE DEDENT
j = i + 1 NEW_LINE while j < len ( arr ) and arr [ j ] == " D " : NEW_LINE INDENT noOfNextD += 1 NEW_LINE j += 1 NEW_LINE DEDENT
curr_max = noOfNextD + 2 NEW_LINE
print ( " " , curr_max , curr_max - 1 , end = " " ) NEW_LINE
last_entry = curr_max - 1 NEW_LINE else : NEW_LINE
print ( " " , last_entry - 1 , end = " " ) NEW_LINE last_entry -= 1 NEW_LINE i += 1 NEW_LINE print ( ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT PrintMinNumberForPattern ( " IDID " ) NEW_LINE PrintMinNumberForPattern ( " I " ) NEW_LINE PrintMinNumberForPattern ( " DD " ) NEW_LINE PrintMinNumberForPattern ( " II " ) NEW_LINE PrintMinNumberForPattern ( " DIDI " ) NEW_LINE PrintMinNumberForPattern ( " IIDDD " ) NEW_LINE PrintMinNumberForPattern ( " DDIDDIID " ) NEW_LINE DEDENT
def printLeast ( arr ) : NEW_LINE
min_avail = 1 NEW_LINE pos_of_I = 0 NEW_LINE
v = [ ] NEW_LINE
if ( arr [ 0 ] == ' I ' ) : NEW_LINE INDENT v . append ( 1 ) NEW_LINE v . append ( 2 ) NEW_LINE min_avail = 3 NEW_LINE pos_of_I = 1 NEW_LINE DEDENT else : NEW_LINE INDENT v . append ( 2 ) NEW_LINE v . append ( 1 ) NEW_LINE min_avail = 3 NEW_LINE pos_of_I = 0 NEW_LINE DEDENT
for i in range ( 1 , len ( arr ) ) : NEW_LINE INDENT if ( arr [ i ] == ' I ' ) : NEW_LINE INDENT v . append ( min_avail ) NEW_LINE min_avail += 1 NEW_LINE pos_of_I = i + 1 NEW_LINE DEDENT else : NEW_LINE INDENT v . append ( v [ i ] ) NEW_LINE for j in range ( pos_of_I , i + 1 ) : NEW_LINE INDENT v [ j ] += 1 NEW_LINE DEDENT min_avail += 1 NEW_LINE DEDENT DEDENT
print ( * v , sep = ' ▁ ' ) NEW_LINE
printLeast ( " IDID " ) NEW_LINE printLeast ( " I " ) NEW_LINE printLeast ( " DD " ) NEW_LINE printLeast ( " II " ) NEW_LINE printLeast ( " DIDI " ) NEW_LINE printLeast ( " IIDDD " ) NEW_LINE printLeast ( " DDIDDIID " ) NEW_LINE
def PrintMinNumberForPattern ( Strr ) : NEW_LINE
res = ' ' NEW_LINE
stack = [ ] NEW_LINE
for i in range ( len ( Strr ) + 1 ) : NEW_LINE
stack . append ( i + 1 ) NEW_LINE
if ( i == len ( Strr ) or Strr [ i ] == ' I ' ) : NEW_LINE
while len ( stack ) > 0 : NEW_LINE
res += str ( stack . pop ( ) ) NEW_LINE res += ' ▁ ' NEW_LINE print ( res ) NEW_LINE
PrintMinNumberForPattern ( " IDID " ) NEW_LINE PrintMinNumberForPattern ( " I " ) NEW_LINE PrintMinNumberForPattern ( " DD " ) NEW_LINE PrintMinNumberForPattern ( " II " ) NEW_LINE PrintMinNumberForPattern ( " DIDI " ) NEW_LINE PrintMinNumberForPattern ( " IIDDD " ) NEW_LINE PrintMinNumberForPattern ( " DDIDDIID " ) NEW_LINE
def getMinNumberForPattern ( seq ) : NEW_LINE INDENT n = len ( seq ) NEW_LINE if ( n >= 9 ) : NEW_LINE INDENT return " - 1" NEW_LINE DEDENT result = [ None ] * ( n + 1 ) NEW_LINE count = 1 NEW_LINE DEDENT
for i in range ( n + 1 ) : NEW_LINE INDENT if ( i == n or seq [ i ] == ' I ' ) : NEW_LINE INDENT for j in range ( i - 1 , - 2 , - 1 ) : NEW_LINE INDENT result [ j + 1 ] = int ( '0' + str ( count ) ) NEW_LINE count += 1 NEW_LINE if ( j >= 0 and seq [ j ] == ' I ' ) : NEW_LINE INDENT break NEW_LINE DEDENT DEDENT DEDENT DEDENT return result NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT inputs = [ " IDID " , " I " , " DD " , " II " , " DIDI " , " IIDDD " , " DDIDDIID " ] NEW_LINE for Input in inputs : NEW_LINE INDENT print ( * ( getMinNumberForPattern ( Input ) ) ) NEW_LINE DEDENT DEDENT
import math as mt NEW_LINE
def isPrime ( n ) : NEW_LINE INDENT i , c = 0 , 0 NEW_LINE for i in range ( 1 , n // 2 ) : NEW_LINE INDENT if ( n % i == 0 ) : NEW_LINE INDENT c += 1 NEW_LINE DEDENT DEDENT if ( c == 1 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT else : NEW_LINE INDENT return 0 NEW_LINE DEDENT DEDENT
def findMinNum ( arr , n ) : NEW_LINE
first , last = 0 , 0 NEW_LINE Hash = [ 0 for i in range ( 10 ) ] NEW_LINE
for i in range ( n ) : NEW_LINE INDENT Hash [ arr [ i ] ] += 1 NEW_LINE DEDENT
print ( " Minimum ▁ number : ▁ " , end = " " ) NEW_LINE for i in range ( 0 , 10 ) : NEW_LINE
for j in range ( Hash [ i ] ) : NEW_LINE INDENT print ( i , end = " " ) NEW_LINE DEDENT print ( ) NEW_LINE
for i in range ( 10 ) : NEW_LINE INDENT if ( Hash [ i ] != 0 ) : NEW_LINE INDENT first = i NEW_LINE break NEW_LINE DEDENT DEDENT
for i in range ( 9 , - 1 , - 1 ) : NEW_LINE INDENT if ( Hash [ i ] != 0 ) : NEW_LINE INDENT last = i NEW_LINE break NEW_LINE DEDENT DEDENT num = first * 10 + last NEW_LINE rev = last * 10 + first NEW_LINE
print ( " Prime ▁ combinations : ▁ " , end = " " ) NEW_LINE if ( isPrime ( num ) and isPrime ( rev ) ) : NEW_LINE INDENT print ( num , " ▁ " , rev ) NEW_LINE DEDENT elif ( isPrime ( num ) ) : NEW_LINE INDENT print ( num ) NEW_LINE DEDENT elif ( isPrime ( rev ) ) : NEW_LINE INDENT print ( rev ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No ▁ combinations ▁ exist " ) NEW_LINE DEDENT
arr = [ 1 , 2 , 4 , 7 , 8 ] NEW_LINE findMinNum ( arr , 5 ) NEW_LINE
def gcd ( a , b ) : NEW_LINE INDENT if ( a == 0 ) : NEW_LINE INDENT return b ; NEW_LINE DEDENT return gcd ( b % a , a ) ; NEW_LINE DEDENT
def coprime ( a , b ) : NEW_LINE
return ( gcd ( a , b ) == 1 ) ; NEW_LINE
def possibleTripletInRange ( L , R ) : NEW_LINE INDENT flag = False ; NEW_LINE possibleA = 0 ; NEW_LINE possibleB = 0 ; NEW_LINE possibleC = 0 ; NEW_LINE DEDENT
for a in range ( L , R + 1 ) : NEW_LINE INDENT for b in range ( a + 1 , R + 1 ) : NEW_LINE INDENT for c in range ( b + 1 , R + 1 ) : NEW_LINE DEDENT DEDENT
if ( coprime ( a , b ) and coprime ( b , c ) and coprime ( a , c ) == False ) : NEW_LINE INDENT flag = True ; NEW_LINE possibleA = a ; NEW_LINE possibleB = b ; NEW_LINE possibleC = c ; NEW_LINE break ; NEW_LINE DEDENT
if ( flag == True ) : NEW_LINE INDENT print ( " ( " , possibleA , " , " , possibleB , " , " , possibleC , " ) ▁ is ▁ one ▁ such " , " possible ▁ triplet ▁ between " , L , " and " , R ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No ▁ Such ▁ Triplet ▁ exists ▁ between " , L , " and " , R ) ; NEW_LINE DEDENT
L = 2 ; NEW_LINE R = 10 ; NEW_LINE possibleTripletInRange ( L , R ) ; NEW_LINE
L = 23 ; NEW_LINE R = 46 ; NEW_LINE possibleTripletInRange ( L , R ) ; NEW_LINE
import numpy as np NEW_LINE
def possibleToReach ( a , b ) : NEW_LINE
c = np . cbrt ( a * b ) NEW_LINE
re1 = a // c NEW_LINE re2 = b // c NEW_LINE
if ( ( re1 * re1 * re2 == a ) and ( re2 * re2 * re1 == b ) ) : NEW_LINE INDENT return True NEW_LINE DEDENT else : NEW_LINE INDENT return False NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT A = 60 NEW_LINE B = 450 NEW_LINE if ( possibleToReach ( A , B ) ) : NEW_LINE INDENT print ( " yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " no " ) NEW_LINE DEDENT DEDENT
def isUndulating ( n ) : NEW_LINE
if ( len ( n ) <= 2 ) : NEW_LINE INDENT return False NEW_LINE DEDENT
for i in range ( 2 , len ( n ) ) : NEW_LINE INDENT if ( n [ i - 2 ] != n [ i ] ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT return True NEW_LINE
n = "1212121" NEW_LINE if ( isUndulating ( n ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
def Series ( n ) : NEW_LINE INDENT sums = 0 NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE INDENT sums += ( i * i ) ; NEW_LINE DEDENT return sums NEW_LINE DEDENT
n = 3 NEW_LINE res = Series ( n ) NEW_LINE print ( res ) NEW_LINE
import math NEW_LINE
def counLastDigitK ( low , high , k ) : NEW_LINE INDENT mlow = 10 * math . ceil ( low / 10.0 ) NEW_LINE mhigh = 10 * int ( high / 10.0 ) NEW_LINE count = ( mhigh - mlow ) / 10 NEW_LINE if ( high % 10 >= k ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT if ( low % 10 <= k and ( low % 10 ) > 0 ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT return int ( count ) NEW_LINE DEDENT
low = 3 NEW_LINE high = 35 NEW_LINE k = 3 NEW_LINE print ( counLastDigitK ( low , high , k ) ) NEW_LINE
sdef sumDivisible ( L , R ) : NEW_LINE
p = int ( R / 6 ) NEW_LINE
q = int ( ( L - 1 ) / 6 ) NEW_LINE
sumR = 3 * ( p * ( p + 1 ) ) NEW_LINE
sumL = ( q * ( q + 1 ) ) * 3 NEW_LINE
return sumR - sumL NEW_LINE
L = 1 NEW_LINE R = 20 NEW_LINE print ( sumDivisible ( L , R ) ) NEW_LINE
import sys NEW_LINE
def prevNum ( string , n ) : NEW_LINE INDENT index = - 1 NEW_LINE DEDENT
for i in range ( n - 2 , - 1 , - 1 ) : NEW_LINE INDENT if int ( string [ i ] ) > int ( string [ i + 1 ] ) : NEW_LINE INDENT index = i NEW_LINE break NEW_LINE DEDENT DEDENT
smallGreatDgt = - 1 NEW_LINE for i in range ( n - 1 , index , - 1 ) : NEW_LINE INDENT if ( smallGreatDgt == - 1 and int ( string [ i ] ) < int ( string [ index ] ) ) : NEW_LINE INDENT smallGreatDgt = i NEW_LINE DEDENT elif ( index > - 1 and int ( string [ i ] ) >= int ( string [ smallGreatDgt ] ) and int ( string [ i ] ) < int ( string [ index ] ) ) : NEW_LINE INDENT smallGreatDgt = i NEW_LINE DEDENT DEDENT
if index == - 1 : NEW_LINE INDENT return " " . join ( " - 1" ) NEW_LINE DEDENT else : NEW_LINE
( string [ index ] , string [ smallGreatDgt ] ) = ( string [ smallGreatDgt ] , string [ index ] ) NEW_LINE return " " . join ( string ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n_str = "34125" NEW_LINE ans = prevNum ( list ( n_str ) , len ( n_str ) ) NEW_LINE print ( ans ) NEW_LINE DEDENT
def horner ( poly , n , x ) : NEW_LINE
result = poly [ 0 ] ; NEW_LINE
for i in range ( 1 , n ) : NEW_LINE INDENT result = ( result * x + poly [ i ] ) ; NEW_LINE DEDENT return result ; NEW_LINE
def findSign ( poly , n , x ) : NEW_LINE INDENT result = horner ( poly , n , x ) ; NEW_LINE if ( result > 0 ) : NEW_LINE INDENT return 1 ; NEW_LINE DEDENT elif ( result < 0 ) : NEW_LINE INDENT return - 1 ; NEW_LINE DEDENT return 0 ; NEW_LINE DEDENT
poly = [ 2 , - 6 , 2 , - 1 ] ; NEW_LINE x = 3 ; NEW_LINE n = len ( poly ) ; NEW_LINE print ( " Sign ▁ of ▁ polynomial ▁ is ▁ " , findSign ( poly , n , x ) ) ; NEW_LINE
isPrime = [ 1 ] * 100005 NEW_LINE
def sieveOfEratostheneses ( ) : NEW_LINE INDENT isPrime [ 1 ] = False NEW_LINE i = 2 NEW_LINE while i * i < 100005 : NEW_LINE INDENT if ( isPrime [ i ] ) : NEW_LINE INDENT j = 2 * i NEW_LINE while j < 100005 : NEW_LINE INDENT isPrime [ j ] = False NEW_LINE j += i NEW_LINE DEDENT DEDENT i += 1 NEW_LINE DEDENT return NEW_LINE DEDENT
def findPrime ( n ) : NEW_LINE INDENT num = n + 1 NEW_LINE DEDENT
while ( num ) : NEW_LINE
if isPrime [ num ] : NEW_LINE INDENT return num NEW_LINE DEDENT
num += 1 NEW_LINE return 0 NEW_LINE
def minNumber ( arr ) : NEW_LINE
sieveOfEratostheneses ( ) NEW_LINE s = 0 NEW_LINE
for i in range ( 0 , len ( arr ) ) : NEW_LINE INDENT s += arr [ i ] NEW_LINE DEDENT
if isPrime [ s ] == True : NEW_LINE INDENT return 0 NEW_LINE DEDENT
num = findPrime ( s ) NEW_LINE
return num - s NEW_LINE
arr = [ 2 , 4 , 6 , 8 , 12 ] NEW_LINE print ( minNumber ( arr ) ) NEW_LINE
def SubArraySum ( arr , n ) : NEW_LINE INDENT temp , result = 0 , 0 NEW_LINE DEDENT
for i in range ( 0 , n ) : NEW_LINE
temp = 0 ; NEW_LINE for j in range ( i , n ) : NEW_LINE
temp += arr [ j ] NEW_LINE result += temp NEW_LINE return result NEW_LINE
arr = [ 1 , 2 , 3 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Sum ▁ of ▁ SubArray ▁ : " , SubArraySum ( arr , n ) ) NEW_LINE
import math NEW_LINE def highestPowerof2 ( n ) : NEW_LINE INDENT p = int ( math . log ( n , 2 ) ) ; NEW_LINE return int ( pow ( 2 , p ) ) ; NEW_LINE DEDENT
n = 10 ; NEW_LINE print ( highestPowerof2 ( n ) ) ; NEW_LINE
' NEW_LINE
def aModM ( s , mod ) : NEW_LINE INDENT number = 0 NEW_LINE for i in range ( len ( s ) ) : NEW_LINE DEDENT
number = ( number * 10 + int ( s [ i ] ) ) NEW_LINE number = number % m NEW_LINE return number NEW_LINE
def ApowBmodM ( a , b , m ) : NEW_LINE
ans = aModM ( a , m ) NEW_LINE mul = ans NEW_LINE
for i in range ( 1 , b ) : NEW_LINE INDENT ans = ( ans * mul ) % m NEW_LINE DEDENT return ans NEW_LINE
a = "987584345091051645734583954832576" NEW_LINE b , m = 3 , 11 NEW_LINE print ApowBmodM ( a , b , m ) NEW_LINE
class Data : NEW_LINE INDENT def __init__ ( self , x , y ) : NEW_LINE INDENT self . x = x NEW_LINE self . y = y NEW_LINE DEDENT DEDENT
def interpolate ( f : list , xi : int , n : int ) -> float : NEW_LINE
result = 0.0 NEW_LINE for i in range ( n ) : NEW_LINE
term = f [ i ] . y NEW_LINE for j in range ( n ) : NEW_LINE INDENT if j != i : NEW_LINE INDENT term = term * ( xi - f [ j ] . x ) / ( f [ i ] . x - f [ j ] . x ) NEW_LINE DEDENT DEDENT
result += term NEW_LINE return result NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
f = [ Data ( 0 , 2 ) , Data ( 1 , 3 ) , Data ( 2 , 12 ) , Data ( 5 , 147 ) ] NEW_LINE
print ( " Value ▁ of ▁ f ( 3 ) ▁ is ▁ : " , interpolate ( f , 3 , 4 ) ) NEW_LINE
def SieveOfSundaram ( n ) : NEW_LINE
nNew = int ( ( n - 1 ) / 2 ) ; NEW_LINE
marked = [ 0 ] * ( nNew + 1 ) ; NEW_LINE
for i in range ( 1 , nNew + 1 ) : NEW_LINE INDENT j = i ; NEW_LINE while ( ( i + j + 2 * i * j ) <= nNew ) : NEW_LINE INDENT marked [ i + j + 2 * i * j ] = 1 ; NEW_LINE j += 1 ; NEW_LINE DEDENT DEDENT
if ( n > 2 ) : NEW_LINE INDENT print ( 2 , end = " ▁ " ) ; NEW_LINE DEDENT
for i in range ( 1 , nNew + 1 ) : NEW_LINE INDENT if ( marked [ i ] == 0 ) : NEW_LINE INDENT print ( ( 2 * i + 1 ) , end = " ▁ " ) ; NEW_LINE DEDENT DEDENT
n = 20 ; NEW_LINE SieveOfSundaram ( n ) ; NEW_LINE
def constructArray ( A , N , K ) : NEW_LINE
B = [ 0 ] * N ; NEW_LINE
totalXOR = A [ 0 ] ^ K ; NEW_LINE
for i in range ( N ) : NEW_LINE INDENT B [ i ] = totalXOR ^ A [ i ] ; NEW_LINE DEDENT
for i in range ( N ) : NEW_LINE INDENT print ( B [ i ] , end = " ▁ " ) ; NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT A = [ 13 , 14 , 10 , 6 ] ; NEW_LINE K = 2 ; NEW_LINE N = len ( A ) ; NEW_LINE DEDENT
constructArray ( A , N , K ) ; NEW_LINE
def extraElement ( A , B , n ) : NEW_LINE
ans = 0 ; NEW_LINE
for i in range ( n ) : NEW_LINE INDENT ans ^= A [ i ] ; NEW_LINE DEDENT for i in range ( n + 1 ) : NEW_LINE INDENT ans ^= B [ i ] ; NEW_LINE DEDENT return ans ; NEW_LINE
A = [ 10 , 15 , 5 ] ; NEW_LINE B = [ 10 , 100 , 15 , 5 ] ; NEW_LINE n = len ( A ) ; NEW_LINE print ( extraElement ( A , B , n ) ) ; NEW_LINE
def hammingDistance ( n1 , n2 ) : NEW_LINE INDENT x = n1 ^ n2 NEW_LINE setBits = 0 NEW_LINE while ( x > 0 ) : NEW_LINE INDENT setBits += x & 1 NEW_LINE x >>= 1 NEW_LINE DEDENT return setBits NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n1 = 9 NEW_LINE n2 = 14 NEW_LINE print ( hammingDistance ( 9 , 14 ) ) NEW_LINE DEDENT
def printSubsets ( n ) : NEW_LINE INDENT for i in range ( n + 1 ) : NEW_LINE INDENT if ( ( n & i ) == i ) : NEW_LINE INDENT print ( i , " ▁ " , end = " " ) NEW_LINE DEDENT DEDENT DEDENT
n = 9 NEW_LINE printSubsets ( n ) NEW_LINE
import math NEW_LINE def setBitNumber ( n ) : NEW_LINE
k = int ( math . log ( n , 2 ) ) NEW_LINE
return 1 << k NEW_LINE
n = 273 NEW_LINE print ( setBitNumber ( n ) ) NEW_LINE
def subset ( ar , n ) : NEW_LINE
res = 0 NEW_LINE
ar . sort ( ) NEW_LINE
for i in range ( 0 , n ) : NEW_LINE INDENT count = 1 NEW_LINE DEDENT
for i in range ( n - 1 ) : NEW_LINE INDENT if ar [ i ] == ar [ i + 1 ] : NEW_LINE INDENT count += 1 NEW_LINE DEDENT else : NEW_LINE INDENT break NEW_LINE DEDENT DEDENT
res = max ( res , count ) NEW_LINE return res NEW_LINE
ar = [ 5 , 6 , 9 , 3 , 4 , 3 , 4 ] NEW_LINE n = len ( ar ) NEW_LINE print ( subset ( ar , n ) ) NEW_LINE
def subset ( arr , n ) : NEW_LINE
mp = { i : 0 for i in range ( 10 ) } NEW_LINE for i in range ( n ) : NEW_LINE INDENT mp [ arr [ i ] ] += 1 NEW_LINE DEDENT
res = 0 NEW_LINE for key , value in mp . items ( ) : NEW_LINE INDENT res = max ( res , value ) NEW_LINE DEDENT return res NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 5 , 6 , 9 , 3 , 4 , 3 , 4 ] NEW_LINE n = len ( arr ) NEW_LINE print ( subset ( arr , n ) ) NEW_LINE DEDENT
psquare = [ ] NEW_LINE
def calcPsquare ( N ) : NEW_LINE INDENT for i in range ( 1 , N ) : NEW_LINE INDENT if i * i > N : NEW_LINE INDENT break NEW_LINE DEDENT psquare . append ( i * i ) NEW_LINE DEDENT DEDENT
def countWays ( index , target ) : NEW_LINE
if ( target == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT if ( index < 0 or target < 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
inc = countWays ( index , target - psquare [ index ] ) NEW_LINE
exc = countWays ( index - 1 , target ) NEW_LINE
return inc + exc NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
N = 9 NEW_LINE
calcPsquare ( N ) NEW_LINE
print ( countWays ( len ( psquare ) - 1 , N ) ) NEW_LINE
sum = 0 NEW_LINE
class TreeNode : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . size = 0 NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def sumofsubtree ( root ) : NEW_LINE
p = [ 1 , 0 ] NEW_LINE
if ( root . left ) : NEW_LINE INDENT ptemp = sumofsubtree ( root . left ) NEW_LINE p [ 1 ] += ptemp [ 0 ] + ptemp [ 1 ] NEW_LINE p [ 0 ] += ptemp [ 0 ] NEW_LINE DEDENT
if ( root . right ) : NEW_LINE INDENT ptemp = sumofsubtree ( root . right ) NEW_LINE p [ 1 ] += ptemp [ 0 ] + ptemp [ 1 ] NEW_LINE p [ 0 ] += ptemp [ 0 ] NEW_LINE DEDENT
root . size = p [ 0 ] NEW_LINE return p NEW_LINE
def distance ( root , target , distancesum , n ) : NEW_LINE INDENT global sum NEW_LINE DEDENT
if ( root . data == target ) : NEW_LINE INDENT sum = distancesum NEW_LINE DEDENT
if ( root . left ) : NEW_LINE
tempsum = ( distancesum - root . left . size + ( n - root . left . size ) ) NEW_LINE
distance ( root . left , target , tempsum , n ) NEW_LINE
if ( root . right ) : NEW_LINE
tempsum = ( distancesum - root . right . size + ( n - root . right . size ) ) NEW_LINE
distance ( root . right , target , tempsum , n ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
root = TreeNode ( 1 ) NEW_LINE root . left = TreeNode ( 2 ) NEW_LINE root . right = TreeNode ( 3 ) NEW_LINE root . left . left = TreeNode ( 4 ) NEW_LINE root . left . right = TreeNode ( 5 ) NEW_LINE root . right . left = TreeNode ( 6 ) NEW_LINE root . right . right = TreeNode ( 7 ) NEW_LINE root . left . left . left = TreeNode ( 8 ) NEW_LINE root . left . left . right = TreeNode ( 9 ) NEW_LINE target = 3 NEW_LINE p = sumofsubtree ( root ) NEW_LINE
totalnodes = p [ 0 ] NEW_LINE distance ( root , target , p [ 1 ] , totalnodes ) NEW_LINE
print ( sum ) NEW_LINE
def rearrangeArray ( A , B , N , K ) : NEW_LINE
B . sort ( reverse = True ) NEW_LINE flag = True NEW_LINE for i in range ( N ) : NEW_LINE
if ( A [ i ] + B [ i ] > K ) : NEW_LINE INDENT flag = False NEW_LINE break NEW_LINE DEDENT if ( flag == False ) : NEW_LINE print ( " - 1" ) NEW_LINE else : NEW_LINE
for i in range ( N ) : NEW_LINE INDENT print ( B [ i ] , end = " ▁ " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE
A = [ 1 , 2 , 3 , 4 , 2 ] NEW_LINE B = [ 1 , 2 , 3 , 1 , 1 ] NEW_LINE N = len ( A ) NEW_LINE K = 5 ; NEW_LINE rearrangeArray ( A , B , N , K ) NEW_LINE
def countRows ( mat ) : NEW_LINE
n = len ( mat ) NEW_LINE m = len ( mat [ 0 ] ) NEW_LINE
count = 0 NEW_LINE
totalSum = 0 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT for j in range ( m ) : NEW_LINE INDENT totalSum += mat [ i ] [ j ] NEW_LINE DEDENT DEDENT
for i in range ( n ) : NEW_LINE
currSum = 0 NEW_LINE
for j in range ( m ) : NEW_LINE INDENT currSum += mat [ i ] [ j ] NEW_LINE DEDENT
if ( currSum > totalSum - currSum ) : NEW_LINE
count += 1 NEW_LINE
print ( count ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
mat = [ [ 2 , - 1 , 5 ] , [ - 3 , 0 , - 2 ] , [ 5 , 1 , 2 ] ] NEW_LINE
countRows ( mat ) NEW_LINE
def areElementsContiguous ( arr , n ) : NEW_LINE
arr . sort ( ) NEW_LINE
for i in range ( 1 , n ) : NEW_LINE INDENT if ( arr [ i ] - arr [ i - 1 ] > 1 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT DEDENT return 1 NEW_LINE
arr = [ 5 , 2 , 3 , 6 , 4 , 4 , 6 , 6 ] NEW_LINE n = len ( arr ) NEW_LINE if areElementsContiguous ( arr , n ) : print ( " Yes " ) NEW_LINE else : print ( " No " ) NEW_LINE
def areElementsContiguous ( arr , n ) : NEW_LINE
max1 = max ( arr ) NEW_LINE min1 = min ( arr ) NEW_LINE m = max1 - min1 + 1 NEW_LINE
if ( m > n ) : NEW_LINE INDENT return False NEW_LINE DEDENT
visited = [ 0 ] * m NEW_LINE
for i in range ( 0 , n ) : NEW_LINE INDENT visited [ arr [ i ] - min1 ] = True NEW_LINE DEDENT
for i in range ( 0 , m ) : NEW_LINE INDENT if ( visited [ i ] == False ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT return True NEW_LINE
arr = [ 5 , 2 , 3 , 6 , 4 , 4 , 6 , 6 ] NEW_LINE n = len ( arr ) NEW_LINE if ( areElementsContiguous ( arr , n ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
def areElementsContiguous ( arr ) : NEW_LINE
us = set ( ) NEW_LINE for i in arr : us . add ( i ) NEW_LINE
count = 1 NEW_LINE
curr_ele = arr [ 0 ] - 1 NEW_LINE
while curr_ele in us : NEW_LINE
count += 1 NEW_LINE
curr_ele -= 1 NEW_LINE
curr_ele = arr [ 0 ] + 1 NEW_LINE
while curr_ele in us : NEW_LINE
count += 1 NEW_LINE
curr_ele += 1 NEW_LINE
return ( count == len ( us ) ) NEW_LINE
arr = [ 5 , 2 , 3 , 6 , 4 , 4 , 6 , 6 ] NEW_LINE if areElementsContiguous ( arr ) : print ( " Yes " ) NEW_LINE else : print ( " No " ) NEW_LINE
import collections NEW_LINE def longest ( a , n , k ) : NEW_LINE INDENT freq = collections . defaultdict ( int ) NEW_LINE start = 0 NEW_LINE end = 0 NEW_LINE now = 0 NEW_LINE l = 0 NEW_LINE for i in range ( n ) : NEW_LINE DEDENT
freq [ a [ i ] ] += 1 NEW_LINE
if ( freq [ a [ i ] ] == 1 ) : NEW_LINE INDENT now += 1 NEW_LINE DEDENT
while ( now > k ) : NEW_LINE
freq [ a [ l ] ] -= 1 NEW_LINE
if ( freq [ a [ l ] ] == 0 ) : NEW_LINE INDENT now -= 1 NEW_LINE DEDENT
l += 1 NEW_LINE
if ( i - l + 1 >= end - start + 1 ) : NEW_LINE INDENT end = i NEW_LINE start = l NEW_LINE DEDENT
for i in range ( start , end + 1 ) : NEW_LINE INDENT print ( a [ i ] , end = " ▁ " ) NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = [ 6 , 5 , 1 , 2 , 3 , 2 , 1 , 4 , 5 ] NEW_LINE n = len ( a ) NEW_LINE k = 3 NEW_LINE longest ( a , n , k ) NEW_LINE DEDENT
def kOverlap ( pairs : list , k ) : NEW_LINE
vec = list ( ) NEW_LINE for i in range ( len ( pairs ) ) : NEW_LINE
vec . append ( ( pairs [ 0 ] , - 1 ) ) NEW_LINE vec . append ( ( pairs [ 1 ] , 1 ) ) NEW_LINE
vec . sort ( key = lambda a : a [ 0 ] ) NEW_LINE
st = list ( ) NEW_LINE for i in range ( len ( vec ) ) : NEW_LINE
cur = vec [ i ] NEW_LINE
if cur [ 1 ] == - 1 : NEW_LINE
st . append ( cur ) NEW_LINE
else : NEW_LINE
st . pop ( ) NEW_LINE
if len ( st ) >= k : NEW_LINE INDENT return True NEW_LINE DEDENT return False NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT pairs = list ( ) NEW_LINE pairs . append ( ( 1 , 3 ) ) NEW_LINE pairs . append ( ( 2 , 4 ) ) NEW_LINE pairs . append ( ( 3 , 5 ) ) NEW_LINE pairs . append ( ( 7 , 10 ) ) NEW_LINE n = len ( pairs ) NEW_LINE k = 3 NEW_LINE if kOverlap ( pairs , k ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT DEDENT
N = 5 NEW_LINE
ptr = [ 0 for i in range ( 501 ) ] NEW_LINE
def findSmallestRange ( arr , n , k ) : NEW_LINE INDENT i , minval , maxval , minrange , minel , maxel , flag , minind = 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 NEW_LINE DEDENT
for i in range ( k + 1 ) : NEW_LINE INDENT ptr [ i ] = 0 NEW_LINE DEDENT minrange = 10 ** 9 NEW_LINE while ( 1 ) : NEW_LINE
minind = - 1 NEW_LINE minval = 10 ** 9 NEW_LINE maxval = - 10 ** 9 NEW_LINE flag = 0 NEW_LINE
for i in range ( k ) : NEW_LINE
if ( ptr [ i ] == n ) : NEW_LINE INDENT flag = 1 NEW_LINE break NEW_LINE DEDENT
if ( ptr [ i ] < n and arr [ i ] [ ptr [ i ] ] < minval ) : NEW_LINE
minind = i NEW_LINE minval = arr [ i ] [ ptr [ i ] ] NEW_LINE
if ( ptr [ i ] < n and arr [ i ] [ ptr [ i ] ] > maxval ) : NEW_LINE INDENT maxval = arr [ i ] [ ptr [ i ] ] NEW_LINE DEDENT
if ( flag ) : NEW_LINE INDENT break NEW_LINE DEDENT ptr [ minind ] += 1 NEW_LINE
if ( ( maxval - minval ) < minrange ) : NEW_LINE INDENT minel = minval NEW_LINE maxel = maxval NEW_LINE minrange = maxel - minel NEW_LINE DEDENT print ( " The ▁ smallest ▁ range ▁ is ▁ [ " , minel , maxel , " ] " ) NEW_LINE
arr = [ [ 4 , 7 , 9 , 12 , 15 ] , [ 0 , 8 , 10 , 14 , 20 ] , [ 6 , 12 , 16 , 30 , 50 ] ] NEW_LINE k = len ( arr ) NEW_LINE findSmallestRange ( arr , N , k ) NEW_LINE
def findLargestd ( S , n ) : NEW_LINE INDENT found = False NEW_LINE DEDENT
S . sort ( ) NEW_LINE
for i in range ( n - 1 , - 1 , - 1 ) : NEW_LINE INDENT for j in range ( 0 , n ) : NEW_LINE DEDENT
if ( i == j ) : NEW_LINE INDENT continue NEW_LINE DEDENT for k in range ( j + 1 , n ) : NEW_LINE INDENT if ( i == k ) : NEW_LINE INDENT continue NEW_LINE DEDENT for l in range ( k + 1 , n ) : NEW_LINE INDENT if ( i == l ) : NEW_LINE INDENT continue NEW_LINE DEDENT DEDENT DEDENT
if ( S [ i ] == S [ j ] + S [ k ] + S [ l ] ) : NEW_LINE INDENT found = True NEW_LINE return S [ i ] NEW_LINE DEDENT if ( found == False ) : NEW_LINE return - 1 NEW_LINE
S = [ 2 , 3 , 5 , 7 , 12 ] NEW_LINE n = len ( S ) NEW_LINE ans = findLargestd ( S , n ) NEW_LINE if ( ans == - 1 ) : NEW_LINE INDENT print ( " No ▁ Solution " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Largest ▁ d ▁ such ▁ that ▁ a ▁ + ▁ b ▁ + " , " c ▁ = ▁ d ▁ is " , ans ) NEW_LINE DEDENT
def findFourElements ( arr , n ) : NEW_LINE INDENT mp = dict ( ) NEW_LINE DEDENT
for i in range ( n - 1 ) : NEW_LINE INDENT for j in range ( i + 1 , n ) : NEW_LINE INDENT mp [ arr [ i ] + arr [ j ] ] = ( i , j ) NEW_LINE DEDENT DEDENT
d = - 10 ** 9 NEW_LINE for i in range ( n - 1 ) : NEW_LINE INDENT for j in range ( i + 1 , n ) : NEW_LINE INDENT abs_diff = abs ( arr [ i ] - arr [ j ] ) NEW_LINE DEDENT DEDENT
if abs_diff in mp . keys ( ) : NEW_LINE
p = mp [ abs_diff ] NEW_LINE if ( p [ 0 ] != i and p [ 0 ] != j and p [ 1 ] != i and p [ 1 ] != j ) : NEW_LINE INDENT d = max ( d , max ( arr [ i ] , arr [ j ] ) ) NEW_LINE DEDENT return d NEW_LINE
arr = [ 2 , 3 , 5 , 7 , 12 ] NEW_LINE n = len ( arr ) NEW_LINE res = findFourElements ( arr , n ) NEW_LINE if ( res == - 10 ** 9 ) : NEW_LINE INDENT print ( " No ▁ Solution . " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( res ) NEW_LINE DEDENT
def CountMaximum ( arr , n , k ) : NEW_LINE
arr . sort ( ) NEW_LINE Sum , count = 0 , 0 NEW_LINE
for i in range ( 0 , n ) : NEW_LINE
Sum += arr [ i ] NEW_LINE
if ( Sum > k ) : NEW_LINE INDENT break NEW_LINE DEDENT
count += 1 NEW_LINE
return count NEW_LINE
arr = [ 30 , 30 , 10 , 10 ] NEW_LINE n = len ( arr ) NEW_LINE k = 50 NEW_LINE
print ( CountMaximum ( arr , n , k ) ) NEW_LINE
def leftRotatebyOne ( arr , n ) : NEW_LINE INDENT temp = arr [ 0 ] NEW_LINE for i in range ( n - 1 ) : NEW_LINE INDENT arr [ i ] = arr [ i + 1 ] NEW_LINE DEDENT arr [ n - 1 ] = temp NEW_LINE DEDENT
def leftRotate ( arr , d , n ) : NEW_LINE INDENT for i in range ( d ) : NEW_LINE INDENT leftRotatebyOne ( arr , n ) NEW_LINE DEDENT DEDENT
def printArray ( arr , size ) : NEW_LINE INDENT for i in range ( size ) : NEW_LINE INDENT print ( " % ▁ d " % arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT DEDENT
arr = [ 1 , 2 , 3 , 4 , 5 , 6 , 7 ] NEW_LINE leftRotate ( arr , 2 , 7 ) NEW_LINE printArray ( arr , 7 ) NEW_LINE
def partSort ( arr , N , a , b ) : NEW_LINE
l = min ( a , b ) NEW_LINE r = max ( a , b ) NEW_LINE
temp = [ 0 for i in range ( r - l + 1 ) ] NEW_LINE j = 0 NEW_LINE for i in range ( l , r + 1 , 1 ) : NEW_LINE INDENT temp [ j ] = arr [ i ] NEW_LINE j += 1 NEW_LINE DEDENT
temp . sort ( reverse = False ) NEW_LINE
j = 0 NEW_LINE for i in range ( l , r + 1 , 1 ) : NEW_LINE INDENT arr [ i ] = temp [ j ] NEW_LINE j += 1 NEW_LINE DEDENT
for i in range ( 0 , N , 1 ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 7 , 8 , 4 , 5 , 2 ] NEW_LINE a = 1 NEW_LINE b = 4 NEW_LINE DEDENT
N = len ( arr ) NEW_LINE partSort ( arr , N , a , b ) NEW_LINE
MAX_SIZE = 10 NEW_LINE
def sortByRow ( mat , n , descending ) : NEW_LINE INDENT for i in range ( n ) : NEW_LINE INDENT if ( descending == True ) : NEW_LINE INDENT mat [ i ] . sort ( reverse = True ) NEW_LINE DEDENT else : NEW_LINE INDENT mat [ i ] . sort ( ) NEW_LINE DEDENT DEDENT DEDENT
def transpose ( mat , n ) : NEW_LINE INDENT for i in range ( n ) : NEW_LINE INDENT for j in range ( i + 1 , n ) : NEW_LINE DEDENT DEDENT
mat [ i ] [ j ] , mat [ j ] [ i ] = mat [ j ] [ i ] , mat [ i ] [ j ] NEW_LINE
def sortMatRowAndColWise ( mat , n ) : NEW_LINE
sortByRow ( mat , n , True ) NEW_LINE
transpose ( mat , n ) NEW_LINE
sortByRow ( mat , n , False ) NEW_LINE
transpose ( mat , n ) ; NEW_LINE
def printMat ( mat , n ) : NEW_LINE INDENT for i in range ( n ) : NEW_LINE INDENT for j in range ( n ) : NEW_LINE INDENT print ( mat [ i ] [ j ] , end = " ▁ " ) NEW_LINE DEDENT print ( ) NEW_LINE DEDENT DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 3 NEW_LINE mat = [ [ 3 , 2 , 1 ] , [ 9 , 8 , 7 ] , [ 6 , 5 , 4 ] ] NEW_LINE print ( " Original ▁ Matrix : ▁ " ) NEW_LINE printMat ( mat , n ) NEW_LINE sortMatRowAndColWise ( mat , n ) NEW_LINE print ( " Matrix ▁ After ▁ Sorting : " ) NEW_LINE printMat ( mat , n ) NEW_LINE DEDENT
def pushZerosToEnd ( arr , n ) : NEW_LINE
count = 0 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT if arr [ i ] != 0 : NEW_LINE DEDENT
arr [ count ] = arr [ i ] NEW_LINE count += 1 NEW_LINE
while count < n : NEW_LINE INDENT arr [ count ] = 0 NEW_LINE count += 1 NEW_LINE DEDENT
arr = [ 1 , 9 , 8 , 4 , 0 , 0 , 2 , 7 , 0 , 6 , 0 , 9 ] NEW_LINE n = len ( arr ) NEW_LINE pushZerosToEnd ( arr , n ) NEW_LINE print ( " Array ▁ after ▁ pushing ▁ all ▁ zeros ▁ to ▁ end ▁ of ▁ array : " ) NEW_LINE print ( arr ) NEW_LINE
def moveZerosToEnd ( arr , n ) : NEW_LINE
count = 0 ; NEW_LINE
for i in range ( 0 , n ) : NEW_LINE INDENT if ( arr [ i ] != 0 ) : NEW_LINE INDENT arr [ count ] , arr [ i ] = arr [ i ] , arr [ count ] NEW_LINE count += 1 NEW_LINE DEDENT DEDENT
def printArray ( arr , n ) : NEW_LINE INDENT for i in range ( 0 , n ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT DEDENT
arr = [ 0 , 1 , 9 , 8 , 4 , 0 , 0 , 2 , 7 , 0 , 6 , 0 , 9 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Original ▁ array : " , end = " ▁ " ) NEW_LINE printArray ( arr , n ) NEW_LINE moveZerosToEnd ( arr , n ) NEW_LINE print ( " Modified array : " , ▁ end = "   " ) NEW_LINE printArray ( arr , n ) NEW_LINE
def pushZerosToEnd ( arr , n ) : NEW_LINE
count = 0 NEW_LINE
for i in range ( 0 , n ) : NEW_LINE INDENT if arr [ i ] != 0 : NEW_LINE DEDENT
arr [ count ] = arr [ i ] NEW_LINE count += 1 NEW_LINE
while ( count < n ) : NEW_LINE INDENT arr [ count ] = 0 NEW_LINE count += 1 NEW_LINE DEDENT
def modifyAndRearrangeArr ( ar , n ) : NEW_LINE
if n == 1 : NEW_LINE INDENT return NEW_LINE DEDENT
for i in range ( 0 , n - 1 ) : NEW_LINE
if ( arr [ i ] != 0 ) and ( arr [ i ] == arr [ i + 1 ] ) : NEW_LINE
arr [ i ] = 2 * arr [ i ] NEW_LINE
arr [ i + 1 ] = 0 NEW_LINE
i += 1 NEW_LINE
pushZerosToEnd ( arr , n ) NEW_LINE
def printArray ( arr , n ) : NEW_LINE INDENT for i in range ( 0 , n ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT DEDENT
arr = [ 0 , 2 , 2 , 2 , 0 , 6 , 6 , 0 , 0 , 8 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Original ▁ array : " , end = " ▁ " ) NEW_LINE printArray ( arr , n ) NEW_LINE modifyAndRearrangeArr ( arr , n ) NEW_LINE print ( " Modified array : " , end = "   " ) NEW_LINE printArray ( arr , n ) NEW_LINE
def shiftAllZeroToLeft ( arr , n ) : NEW_LINE
lastSeenNonZero = 0 NEW_LINE for index in range ( 0 , n ) : NEW_LINE
if ( array [ index ] != 0 ) : NEW_LINE
array [ index ] , array [ lastSeenNonZero ] = array [ lastSeenNonZero ] , array [ index ] NEW_LINE
lastSeenNonZero += 1 NEW_LINE
def printArray ( arr , n ) : NEW_LINE INDENT for i in range ( n ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT print ( ) NEW_LINE DEDENT
def RearrangePosNeg ( arr , n ) : NEW_LINE INDENT for i in range ( 1 , n ) : NEW_LINE INDENT key = arr [ i ] NEW_LINE DEDENT DEDENT
if ( key > 0 ) : NEW_LINE INDENT continue NEW_LINE DEDENT
j = i - 1 NEW_LINE while ( j >= 0 and arr [ j ] > 0 ) : NEW_LINE INDENT arr [ j + 1 ] = arr [ j ] NEW_LINE j = j - 1 NEW_LINE DEDENT
arr [ j + 1 ] = key NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ - 12 , 11 , - 13 , - 5 , 6 , - 7 , 5 , - 3 , - 6 ] NEW_LINE n = len ( arr ) NEW_LINE RearrangePosNeg ( arr , n ) NEW_LINE printArray ( arr , n ) NEW_LINE DEDENT
def printArray ( A , size ) : NEW_LINE INDENT for i in range ( 0 , size ) : NEW_LINE INDENT print ( A [ i ] , end = " ▁ " ) NEW_LINE DEDENT print ( ) NEW_LINE DEDENT
def reverse ( arr , l , r ) : NEW_LINE INDENT if l < r : NEW_LINE INDENT arr [ l ] , arr [ r ] = arr [ r ] , arr [ l ] NEW_LINE l , r = l + 1 , r - 1 NEW_LINE reverse ( arr , l , r ) NEW_LINE DEDENT DEDENT
def merge ( arr , l , m , r ) : NEW_LINE
i = l NEW_LINE
j = m + 1 NEW_LINE while i <= m and arr [ i ] < 0 : NEW_LINE INDENT i += 1 NEW_LINE DEDENT
while j <= r and arr [ j ] < 0 : NEW_LINE INDENT j += 1 NEW_LINE DEDENT
reverse ( arr , i , m ) NEW_LINE
reverse ( arr , m + 1 , j - 1 ) NEW_LINE
reverse ( arr , i , j - 1 ) NEW_LINE
def RearrangePosNeg ( arr , l , r ) : NEW_LINE INDENT if l < r : NEW_LINE DEDENT
m = l + ( r - l ) // 2 NEW_LINE
RearrangePosNeg ( arr , l , m ) NEW_LINE RearrangePosNeg ( arr , m + 1 , r ) NEW_LINE merge ( arr , l , m , r ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ - 12 , 11 , - 13 , - 5 , 6 , - 7 , 5 , - 3 , - 6 ] NEW_LINE arr_size = len ( arr ) NEW_LINE RearrangePosNeg ( arr , 0 , arr_size - 1 ) NEW_LINE printArray ( arr , arr_size ) NEW_LINE DEDENT
def RearrangePosNeg ( arr , n ) : NEW_LINE INDENT i = 0 NEW_LINE j = n - 1 NEW_LINE while ( True ) : NEW_LINE DEDENT
while ( arr [ i ] < 0 and i < n ) : NEW_LINE INDENT i += 1 NEW_LINE DEDENT
while ( arr [ j ] > 0 and j >= 0 ) : NEW_LINE INDENT j -= 1 NEW_LINE DEDENT
if ( i < j ) : NEW_LINE INDENT arr [ i ] , arr [ j ] = arr [ j ] , arr [ i ] NEW_LINE DEDENT else : NEW_LINE INDENT break NEW_LINE DEDENT
arr = [ - 12 , 11 , - 13 , - 5 , 6 , - 7 , 5 , - 3 , - 6 ] NEW_LINE n = len ( arr ) NEW_LINE RearrangePosNeg ( arr , n ) NEW_LINE print ( * arr ) NEW_LINE
def winner ( arr , N ) : NEW_LINE
if ( N % 2 == 1 ) : NEW_LINE INDENT print ( " A " ) NEW_LINE DEDENT
else : NEW_LINE INDENT print ( " B " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE
arr = [ 24 , 45 , 45 , 24 ] NEW_LINE
N = len ( arr ) NEW_LINE winner ( arr , N ) NEW_LINE
import math NEW_LINE sz = 20 NEW_LINE sqr = int ( math . sqrt ( sz ) ) + 1 NEW_LINE
def precomputeExpressionForAllVal ( arr , N , dp ) : NEW_LINE
for i in range ( N - 1 , - 1 , - 1 ) : NEW_LINE
for j in range ( 1 , int ( math . sqrt ( N ) ) + 1 ) : NEW_LINE
if ( i + j < N ) : NEW_LINE
dp [ i ] [ j ] = arr [ i ] + dp [ i + j ] [ j ] NEW_LINE else : NEW_LINE
dp [ i ] [ j ] = arr [ i ] NEW_LINE
def querySum ( arr , N , Q , M ) : NEW_LINE
dp = [ [ 0 for x in range ( sz ) ] for x in range ( sqr ) ] NEW_LINE precomputeExpressionForAllVal ( arr , N , dp ) NEW_LINE
for i in range ( 0 , M ) : NEW_LINE INDENT x = Q [ i ] [ 0 ] NEW_LINE y = Q [ i ] [ 1 ] NEW_LINE DEDENT
if ( y <= math . sqrt ( N ) ) : NEW_LINE INDENT print ( dp [ x ] [ y ] ) NEW_LINE continue NEW_LINE DEDENT
sum = 0 NEW_LINE
while ( x < N ) : NEW_LINE
sum += arr [ x ] NEW_LINE
x += y NEW_LINE print ( sum ) NEW_LINE
arr = [ 1 , 2 , 7 , 5 , 4 ] NEW_LINE Q = [ [ 2 , 1 ] , [ 3 , 2 ] ] NEW_LINE N = len ( arr ) NEW_LINE M = len ( Q [ 0 ] ) NEW_LINE querySum ( arr , N , Q , M ) NEW_LINE
def findElements ( arr , n ) : NEW_LINE
for i in range ( n ) : NEW_LINE INDENT count = 0 NEW_LINE for j in range ( 0 , n ) : NEW_LINE INDENT if arr [ j ] > arr [ i ] : NEW_LINE INDENT count = count + 1 NEW_LINE DEDENT DEDENT if count >= 2 : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT DEDENT
arr = [ 2 , - 6 , 3 , 5 , 1 ] NEW_LINE n = len ( arr ) NEW_LINE findElements ( arr , n ) NEW_LINE
def findElements ( arr , n ) : NEW_LINE INDENT arr . sort ( ) NEW_LINE for i in range ( 0 , n - 2 ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT DEDENT
arr = [ 2 , - 6 , 3 , 5 , 1 ] NEW_LINE n = len ( arr ) NEW_LINE findElements ( arr , n ) NEW_LINE
import sys NEW_LINE def findElements ( arr , n ) : NEW_LINE INDENT first = - sys . maxsize NEW_LINE second = - sys . maxsize NEW_LINE for i in range ( 0 , n ) : NEW_LINE DEDENT
if ( arr [ i ] > first ) : NEW_LINE INDENT second = first NEW_LINE first = arr [ i ] NEW_LINE DEDENT
elif ( arr [ i ] > second ) : NEW_LINE INDENT second = arr [ i ] NEW_LINE DEDENT for i in range ( 0 , n ) : NEW_LINE if ( arr [ i ] < second ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT
arr = [ 2 , - 6 , 3 , 5 , 1 ] NEW_LINE n = len ( arr ) NEW_LINE findElements ( arr , n ) NEW_LINE
def getMinOps ( arr ) : NEW_LINE
res = 0 NEW_LINE for i in range ( len ( arr ) - 1 ) : NEW_LINE
res += max ( arr [ i + 1 ] - arr [ i ] , 0 ) NEW_LINE
return res NEW_LINE
arr = [ 1 , 3 , 4 , 1 , 2 ] NEW_LINE print ( getMinOps ( arr ) ) NEW_LINE
def findFirstMissing ( array , start , end ) : NEW_LINE INDENT if ( start > end ) : NEW_LINE INDENT return end + 1 NEW_LINE DEDENT if ( start != array [ start ] ) : NEW_LINE INDENT return start ; NEW_LINE DEDENT mid = int ( ( start + end ) / 2 ) NEW_LINE DEDENT
if ( array [ mid ] == mid ) : NEW_LINE INDENT return findFirstMissing ( array , mid + 1 , end ) NEW_LINE DEDENT return findFirstMissing ( array , start , mid ) NEW_LINE
arr = [ 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 10 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Smallest ▁ missing ▁ element ▁ is " , findFirstMissing ( arr , 0 , n - 1 ) ) NEW_LINE
def findFirstMissing ( arr , start , end , first ) : NEW_LINE INDENT if ( start < end ) : NEW_LINE INDENT mid = int ( ( start + end ) / 2 ) NEW_LINE DEDENT DEDENT
if ( arr [ mid ] != mid + first ) : NEW_LINE INDENT return findFirstMissing ( arr , start , mid , first ) NEW_LINE DEDENT else : NEW_LINE INDENT return findFirstMissing ( arr , mid + 1 , end , first ) NEW_LINE DEDENT return start + first NEW_LINE
def findSmallestMissinginSortedArray ( arr ) : NEW_LINE
if ( arr [ 0 ] != 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
if ( arr [ - 1 ] == len ( arr ) - 1 ) : NEW_LINE INDENT return len ( arr ) NEW_LINE DEDENT first = arr [ 0 ] NEW_LINE return findFirstMissing ( arr , 0 , len ( arr ) - 1 , first ) NEW_LINE
arr = [ 0 , 1 , 2 , 3 , 4 , 5 , 7 ] NEW_LINE n = len ( arr ) NEW_LINE
print ( " First ▁ Missing ▁ element ▁ is ▁ : " , findSmallestMissinginSortedArray ( arr ) ) NEW_LINE
def find_max_sum ( arr ) : NEW_LINE INDENT incl = 0 NEW_LINE excl = 0 NEW_LINE for i in arr : NEW_LINE DEDENT
new_excl = excl if excl > incl else incl NEW_LINE
incl = excl + i NEW_LINE excl = new_excl NEW_LINE
return ( excl if excl > incl else incl ) NEW_LINE
arr = [ 5 , 5 , 10 , 100 , 10 , 5 ] NEW_LINE print find_max_sum ( arr ) NEW_LINE
def countChanges ( matrix , n , m ) : NEW_LINE
dist = n + m - 1 NEW_LINE
freq = [ [ 0 ] * 10 for i in range ( dist ) ] NEW_LINE
for i in range ( n ) : NEW_LINE INDENT for j in range ( m ) : NEW_LINE DEDENT
freq [ i + j ] [ matrix [ i ] [ j ] ] += 1 NEW_LINE min_changes_sum = 0 NEW_LINE for i in range ( dist // 2 ) : NEW_LINE maximum = 0 NEW_LINE total_values = 0 NEW_LINE
for j in range ( 10 ) : NEW_LINE INDENT maximum = max ( maximum , freq [ i ] [ j ] + freq [ n + m - 2 - i ] [ j ] ) NEW_LINE total_values += ( freq [ i ] [ j ] + freq [ n + m - 2 - i ] [ j ] ) NEW_LINE DEDENT
min_changes_sum += ( total_values - maximum ) NEW_LINE
return min_changes_sum NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
mat = [ [ 1 , 2 ] , [ 3 , 5 ] ] NEW_LINE
print ( countChanges ( mat , 2 , 2 ) ) NEW_LINE
import math NEW_LINE
def buildSparseTable ( arr , n ) : NEW_LINE
for i in range ( 0 , n ) : NEW_LINE INDENT lookup [ i ] [ 0 ] = arr [ i ] NEW_LINE DEDENT j = 1 NEW_LINE
while ( 1 << j ) <= n : NEW_LINE
i = 0 NEW_LINE while ( i + ( 1 << j ) - 1 ) < n : NEW_LINE
if ( lookup [ i ] [ j - 1 ] < lookup [ i + ( 1 << ( j - 1 ) ) ] [ j - 1 ] ) : NEW_LINE INDENT lookup [ i ] [ j ] = lookup [ i ] [ j - 1 ] NEW_LINE DEDENT else : NEW_LINE INDENT lookup [ i ] [ j ] = lookup [ i + ( 1 << ( j - 1 ) ) ] [ j - 1 ] NEW_LINE DEDENT i += 1 NEW_LINE j += 1 NEW_LINE
def query ( L , R ) : NEW_LINE
j = int ( math . log2 ( R - L + 1 ) ) NEW_LINE
if lookup [ L ] [ j ] <= lookup [ R - ( 1 << j ) + 1 ] [ j ] : NEW_LINE INDENT return lookup [ L ] [ j ] NEW_LINE DEDENT else : NEW_LINE INDENT return lookup [ R - ( 1 << j ) + 1 ] [ j ] NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = [ 7 , 2 , 3 , 0 , 5 , 10 , 3 , 12 , 18 ] NEW_LINE n = len ( a ) NEW_LINE MAX = 500 NEW_LINE lookup = [ [ 0 for i in range ( MAX ) ] for j in range ( MAX ) ] NEW_LINE buildSparseTable ( a , n ) NEW_LINE print ( query ( 0 , 4 ) ) NEW_LINE print ( query ( 4 , 7 ) ) NEW_LINE print ( query ( 7 , 8 ) ) NEW_LINE DEDENT
import math NEW_LINE
def buildSparseTable ( arr , n ) : NEW_LINE
for i in range ( 0 , n ) : NEW_LINE INDENT table [ i ] [ 0 ] = arr [ i ] NEW_LINE DEDENT
j = 1 NEW_LINE while ( 1 << j ) <= n : NEW_LINE INDENT i = 0 NEW_LINE while i <= n - ( 1 << j ) : NEW_LINE INDENT table [ i ] [ j ] = math . gcd ( table [ i ] [ j - 1 ] , table [ i + ( 1 << ( j - 1 ) ) ] [ j - 1 ] ) NEW_LINE i += 1 NEW_LINE DEDENT j += 1 NEW_LINE DEDENT
def query ( L , R ) : NEW_LINE
j = int ( math . log2 ( R - L + 1 ) ) NEW_LINE
return math . gcd ( table [ L ] [ j ] , table [ R - ( 1 << j ) + 1 ] [ j ] ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = [ 7 , 2 , 3 , 0 , 5 , 10 , 3 , 12 , 18 ] NEW_LINE n = len ( a ) NEW_LINE MAX = 500 NEW_LINE table = [ [ 0 for i in range ( MAX ) ] for j in range ( MAX ) ] NEW_LINE buildSparseTable ( a , n ) NEW_LINE print ( query ( 0 , 2 ) ) NEW_LINE print ( query ( 1 , 3 ) ) NEW_LINE print ( query ( 4 , 5 ) ) NEW_LINE DEDENT
def minimizeWithKSwaps ( arr , n , k ) : NEW_LINE INDENT for i in range ( n - 1 ) : NEW_LINE DEDENT
pos = i NEW_LINE for j in range ( i + 1 , n ) : NEW_LINE
if ( j - i > k ) : NEW_LINE INDENT break NEW_LINE DEDENT
if ( arr [ j ] < arr [ pos ] ) : NEW_LINE INDENT pos = j NEW_LINE DEDENT
for j in range ( pos , i , - 1 ) : NEW_LINE INDENT arr [ j ] , arr [ j - 1 ] = arr [ j - 1 ] , arr [ j ] NEW_LINE DEDENT
k -= pos - i NEW_LINE
n , k = 5 , 3 NEW_LINE arr = [ 7 , 6 , 9 , 2 , 1 ] NEW_LINE
/ * Function calling * / NEW_LINE minimizeWithKSwaps ( arr , n , k ) NEW_LINE
for i in range ( n ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT
def findMaxAverage ( arr , n , k ) : NEW_LINE
if k > n : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
csum = [ 0 ] * n NEW_LINE csum [ 0 ] = arr [ 0 ] NEW_LINE for i in range ( 1 , n ) : NEW_LINE INDENT csum [ i ] = csum [ i - 1 ] + arr [ i ] ; NEW_LINE DEDENT
max_sum = csum [ k - 1 ] NEW_LINE max_end = k - 1 NEW_LINE
for i in range ( k , n ) : NEW_LINE INDENT curr_sum = csum [ i ] - csum [ i - k ] NEW_LINE if curr_sum > max_sum : NEW_LINE INDENT max_sum = curr_sum NEW_LINE max_end = i NEW_LINE DEDENT DEDENT
return max_end - k + 1 NEW_LINE
arr = [ 1 , 12 , - 5 , - 6 , 50 , 3 ] NEW_LINE k = 4 NEW_LINE n = len ( arr ) NEW_LINE print ( " The ▁ maximum ▁ average ▁ subarray ▁ of ▁ length " , k , " begins ▁ at ▁ index " , findMaxAverage ( arr , n , k ) ) NEW_LINE
def findMaxAverage ( arr , n , k ) : NEW_LINE
if ( k > n ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
sum = arr [ 0 ] NEW_LINE for i in range ( 1 , k ) : NEW_LINE INDENT sum += arr [ i ] NEW_LINE DEDENT max_sum = sum NEW_LINE max_end = k - 1 NEW_LINE
for i in range ( k , n ) : NEW_LINE INDENT sum = sum + arr [ i ] - arr [ i - k ] NEW_LINE if ( sum > max_sum ) : NEW_LINE INDENT max_sum = sum NEW_LINE max_end = i NEW_LINE DEDENT DEDENT
return max_end - k + 1 NEW_LINE
arr = [ 1 , 12 , - 5 , - 6 , 50 , 3 ] NEW_LINE k = 4 NEW_LINE n = len ( arr ) NEW_LINE print ( " The ▁ maximum ▁ average ▁ subarray ▁ of ▁ length " , k , " begins ▁ at ▁ index " , findMaxAverage ( arr , n , k ) ) NEW_LINE
m = dict ( ) NEW_LINE
def findMinimum ( a , n , pos , myturn ) : NEW_LINE
if ( pos , myturn ) in m : NEW_LINE INDENT return m [ ( pos , myturn ) ] ; NEW_LINE DEDENT
if ( pos >= n - 1 ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT
if ( not myturn ) : NEW_LINE
ans = min ( findMinimum ( a , n , pos + 1 , not myturn ) + a [ pos ] , findMinimum ( a , n , pos + 2 , not myturn ) + a [ pos ] + a [ pos + 1 ] ) ; NEW_LINE
m [ ( pos , myturn ) ] = ans ; NEW_LINE
return ans ; NEW_LINE
if ( myturn ) : NEW_LINE
ans = min ( findMinimum ( a , n , pos + 1 , not myturn ) , findMinimum ( a , n , pos + 2 , not myturn ) ) ; NEW_LINE
m [ ( pos , myturn ) ] = ans ; NEW_LINE
return ans ; NEW_LINE return 0 ; NEW_LINE
def countPenality ( arr , N ) : NEW_LINE
pos = 0 ; NEW_LINE
turn = False ; NEW_LINE
return findMinimum ( arr , N , pos , turn ) + 1 ; NEW_LINE
def printAnswer ( arr , N ) : NEW_LINE
a = countPenality ( arr , N ) ; NEW_LINE
sum = 0 ; NEW_LINE for i in range ( N ) : NEW_LINE INDENT sum += arr [ i ] ; NEW_LINE DEDENT
print ( a ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 ] NEW_LINE N = len ( arr ) NEW_LINE printAnswer ( arr , N ) ; NEW_LINE DEDENT
import math NEW_LINE MAX = 1000001 NEW_LINE
prime = [ True ] * MAX NEW_LINE
def SieveOfEratosthenes ( ) : NEW_LINE INDENT p = 2 NEW_LINE while p * p <= MAX : NEW_LINE DEDENT
if prime [ p ] == True : NEW_LINE
for i in range ( p * p , MAX , p ) : NEW_LINE INDENT prime [ i ] = False NEW_LINE DEDENT p += 1 NEW_LINE
def getMid ( s , e ) : NEW_LINE INDENT return s + ( e - s ) // 2 NEW_LINE DEDENT
def getSumUtil ( st , ss , se , qs , qe , si ) : NEW_LINE
if qs <= ss and qe >= se : NEW_LINE INDENT return st [ si ] NEW_LINE DEDENT
if se < qs or ss > qe : NEW_LINE INDENT return 0 NEW_LINE DEDENT
mid = getMid ( ss , se ) NEW_LINE return ( getSumUtil ( st , ss , mid , qs , qe , 2 * si + 1 ) + getSumUtil ( st , mid + 1 , se , qs , qe , 2 * si + 2 ) ) NEW_LINE
def updateValueUtil ( st , ss , se , i , diff , si ) : NEW_LINE
if i < ss or i > se : NEW_LINE INDENT return NEW_LINE DEDENT
st [ si ] = st [ si ] + diff NEW_LINE if se != ss : NEW_LINE INDENT mid = getMid ( ss , se ) NEW_LINE updateValueUtil ( st , ss , mid , i , diff , 2 * si + 1 ) NEW_LINE updateValueUtil ( st , mid + 1 , se , i , diff , 2 * si + 2 ) NEW_LINE DEDENT
def updateValue ( arr , st , n , i , new_val ) : NEW_LINE
if i < 0 or i > n - 1 : NEW_LINE INDENT print ( - 1 ) NEW_LINE return NEW_LINE DEDENT
diff = new_val - arr [ i ] NEW_LINE prev_val = arr [ i ] NEW_LINE
arr [ i ] = new_val NEW_LINE
if prime [ new_val ] or prime [ prev_val ] : NEW_LINE
if not prime [ prev_val ] : NEW_LINE INDENT updateValueUtil ( st , 0 , n - 1 , i , new_val , 0 ) NEW_LINE DEDENT
elif not prime [ new_val ] : NEW_LINE INDENT updateValueUtil ( st , 0 , n - 1 , i , - prev_val , 0 ) NEW_LINE DEDENT
else : NEW_LINE INDENT updateValueUtil ( st , 0 , n - 1 , i , diff , 0 ) NEW_LINE DEDENT
def getSum ( st , n , qs , qe ) : NEW_LINE
if qs < 0 or qe > n - 1 or qs > qe : NEW_LINE INDENT return - 1 NEW_LINE DEDENT return getSumUtil ( st , 0 , n - 1 , qs , qe , 0 ) NEW_LINE
def constructSTUtil ( arr , ss , se , st , si ) : NEW_LINE
if ss == se : NEW_LINE
if prime [ arr [ ss ] ] : NEW_LINE INDENT st [ si ] = arr [ ss ] NEW_LINE DEDENT else : NEW_LINE INDENT st [ si ] = 0 NEW_LINE DEDENT return st [ si ] NEW_LINE
mid = getMid ( ss , se ) NEW_LINE st [ si ] = ( constructSTUtil ( arr , ss , mid , st , 2 * si + 1 ) + constructSTUtil ( arr , mid + 1 , se , st , 2 * si + 2 ) ) NEW_LINE return st [ si ] NEW_LINE
def constructST ( arr , n ) : NEW_LINE
x = int ( math . ceil ( math . log2 ( n ) ) ) NEW_LINE
max_size = 2 * int ( pow ( 2 , x ) ) - 1 NEW_LINE
st = [ 0 ] * max_size NEW_LINE
constructSTUtil ( arr , 0 , n - 1 , st , 0 ) NEW_LINE
return st NEW_LINE
arr = [ 1 , 3 , 5 , 7 , 9 , 11 ] NEW_LINE n = len ( arr ) NEW_LINE Q = [ [ 1 , 1 , 3 ] , [ 2 , 1 , 10 ] , [ 1 , 1 , 3 ] ] NEW_LINE
SieveOfEratosthenes ( ) NEW_LINE
st = constructST ( arr , n ) NEW_LINE
print ( getSum ( st , n , 1 , 3 ) ) NEW_LINE
updateValue ( arr , st , n , 1 , 10 ) NEW_LINE
print ( getSum ( st , n , 1 , 3 ) ) NEW_LINE
mod = 1000000007 NEW_LINE dp = [ [ - 1 for i in range ( 1000 ) ] for j in range ( 1000 ) ] ; NEW_LINE def calculate ( pos , prev , s , index ) : NEW_LINE
if ( pos == len ( s ) ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT
if ( dp [ pos ] [ prev ] != - 1 ) : NEW_LINE INDENT return dp [ pos ] [ prev ] NEW_LINE DEDENT
answer = 0 NEW_LINE for i in range ( len ( index ) ) : NEW_LINE INDENT if ( index [ i ] > prev ) : NEW_LINE INDENT answer = ( answer % mod + calculate ( pos + 1 , index [ i ] , s , index ) % mod ) % mod NEW_LINE DEDENT DEDENT dp [ pos ] [ prev ] = 4 NEW_LINE
return dp [ pos ] [ prev ] NEW_LINE def countWays ( a , s ) : NEW_LINE n = len ( a ) NEW_LINE
index = [ [ ] for i in range ( 26 ) ] NEW_LINE for i in range ( n ) : NEW_LINE INDENT for j in range ( len ( a [ i ] ) ) : NEW_LINE DEDENT
index [ ord ( a [ i ] [ j ] ) - ord ( ' a ' ) ] . append ( j + 1 ) ; NEW_LINE
return calculate ( 0 , 0 , s , index [ 0 ] ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT A = [ ] NEW_LINE A . append ( " adc " ) NEW_LINE A . append ( " aec " ) NEW_LINE A . append ( " erg " ) NEW_LINE S = " ac " NEW_LINE print ( countWays ( A , S ) ) NEW_LINE DEDENT
MAX = 10005 NEW_LINE MOD = 1000000007 NEW_LINE
def countNum ( idx , sum , tight , num , len1 , k ) : NEW_LINE INDENT if ( len1 == idx ) : NEW_LINE INDENT if ( sum == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT else : NEW_LINE INDENT return 0 NEW_LINE DEDENT DEDENT if ( dp [ idx ] [ sum ] [ tight ] != - 1 ) : NEW_LINE INDENT return dp [ idx ] [ sum ] [ tight ] NEW_LINE DEDENT res = 0 NEW_LINE DEDENT
if ( tight == 0 ) : NEW_LINE INDENT limit = num [ idx ] NEW_LINE DEDENT
else : NEW_LINE INDENT limit = 9 NEW_LINE DEDENT for i in range ( limit + 1 ) : NEW_LINE
new_tight = tight NEW_LINE if ( tight == 0 and i < limit ) : NEW_LINE INDENT new_tight = 1 NEW_LINE DEDENT res += countNum ( idx + 1 , ( sum + i ) % k , new_tight , num , len1 , k ) NEW_LINE res %= MOD NEW_LINE
if ( res < 0 ) : NEW_LINE INDENT res += MOD NEW_LINE DEDENT dp [ idx ] [ sum ] [ tight ] = res NEW_LINE return dp [ idx ] [ sum ] [ tight ] NEW_LINE
def process ( s ) : NEW_LINE INDENT num = [ ] NEW_LINE for i in range ( len ( s ) ) : NEW_LINE INDENT num . append ( ord ( s [ i ] ) - ord ( '0' ) ) NEW_LINE DEDENT return num NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE
n = "98765432109876543210" NEW_LINE
len1 = len ( n ) NEW_LINE k = 58 NEW_LINE
dp = [ [ [ - 1 for i in range ( 2 ) ] for j in range ( 101 ) ] for k in range ( MAX ) ] NEW_LINE
num = process ( n ) NEW_LINE print ( countNum ( 0 , 0 , 0 , num , len1 , k ) ) NEW_LINE
def maxWeight ( arr , n , w1_r , w2_r , i ) : NEW_LINE
if i == n : NEW_LINE INDENT return 0 NEW_LINE DEDENT if dp [ i ] [ w1_r ] [ w2_r ] != - 1 : NEW_LINE INDENT return dp [ i ] [ w1_r ] [ w2_r ] NEW_LINE DEDENT
fill_w1 , fill_w2 , fill_none = 0 , 0 , 0 NEW_LINE if w1_r >= arr [ i ] : NEW_LINE INDENT fill_w1 = arr [ i ] + maxWeight ( arr , n , w1_r - arr [ i ] , w2_r , i + 1 ) NEW_LINE DEDENT if w2_r >= arr [ i ] : NEW_LINE INDENT fill_w2 = arr [ i ] + maxWeight ( arr , n , w1_r , w2_r - arr [ i ] , i + 1 ) NEW_LINE DEDENT fill_none = maxWeight ( arr , n , w1_r , w2_r , i + 1 ) NEW_LINE
dp [ i ] [ w1_r ] [ w2_r ] = max ( fill_none , max ( fill_w1 , fill_w2 ) ) NEW_LINE return dp [ i ] [ w1_r ] [ w2_r ] NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
arr = [ 8 , 2 , 3 ] NEW_LINE maxN , maxW = 31 , 31 NEW_LINE
dp = [ [ [ - 1 ] * maxW ] * maxW ] * maxN NEW_LINE
n = len ( arr ) NEW_LINE
w1 , w2 = 10 , 3 NEW_LINE
print ( maxWeight ( arr , n , w1 , w2 , 0 ) ) NEW_LINE
def CountWays ( n ) : NEW_LINE
noOfWays = [ 0 ] * ( n + 1 ) NEW_LINE noOfWays [ 0 ] = 1 NEW_LINE noOfWays [ 1 ] = 1 NEW_LINE noOfWays [ 2 ] = 1 + 1 NEW_LINE
for i in range ( 3 , n + 1 ) : NEW_LINE
noOfWays [ i ] = noOfWays [ 3 - 1 ] NEW_LINE
+ noOfWays [ 3 - 3 ] NEW_LINE
noOfWays [ 0 ] = noOfWays [ 1 ] NEW_LINE noOfWays [ 1 ] = noOfWays [ 2 ] NEW_LINE noOfWays [ 2 ] = noOfWays [ i ] NEW_LINE return noOfWays [ n ] NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 5 NEW_LINE print ( CountWays ( n ) ) NEW_LINE DEDENT
MAX = 105 NEW_LINE def sieve ( ) : NEW_LINE INDENT i = 2 NEW_LINE while ( i * i < MAX ) : NEW_LINE INDENT if ( prime [ i ] == 0 ) : NEW_LINE INDENT for j in range ( i * i , MAX , i ) : NEW_LINE INDENT prime [ j ] = 1 ; NEW_LINE DEDENT DEDENT i += 1 NEW_LINE DEDENT DEDENT
def dfs ( i , j , k , q , n , m ) : NEW_LINE
if ( mappedMatrix [ i ] [ j ] == 0 or i > n or j > m or mark [ i ] [ j ] or q != 0 ) : NEW_LINE INDENT return q ; NEW_LINE DEDENT
mark [ i ] [ j ] = 1 ; NEW_LINE
ans [ k ] = [ i , j ] NEW_LINE
if ( i == n and j == m ) : NEW_LINE
q = k ; NEW_LINE return q ; NEW_LINE
q = dfs ( i + 1 , j + 1 , k + 1 , q , n , m ) ; NEW_LINE
q = dfs ( i + 1 , j , k + 1 , q , n , m ) ; NEW_LINE
q = dfs ( i , j + 1 , k + 1 , q , n , m ) ; NEW_LINE return q NEW_LINE
def lexicographicalPath ( n , m ) : NEW_LINE
q = 0 ; NEW_LINE global ans , mark NEW_LINE
ans = [ [ 0 , 0 ] for i in range ( MAX ) ] NEW_LINE
mark = [ [ 0 for j in range ( MAX ) ] for i in range ( MAX ) ] NEW_LINE
q = dfs ( 1 , 1 , 1 , q , n , m ) ; NEW_LINE
for i in range ( 1 , q + 1 ) : NEW_LINE INDENT print ( str ( ans [ i ] [ 0 ] ) + ' ▁ ' + str ( ans [ i ] [ 1 ] ) ) NEW_LINE DEDENT
def countPrimePath ( n , m ) : NEW_LINE INDENT global dp NEW_LINE dp = [ [ 0 for j in range ( MAX ) ] for i in range ( MAX ) ] NEW_LINE dp [ 1 ] [ 1 ] = 1 ; NEW_LINE DEDENT
for i in range ( 1 , n + 1 ) : NEW_LINE INDENT for j in range ( 1 , m + 1 ) : NEW_LINE DEDENT
if ( i == 1 and j == 1 ) : NEW_LINE INDENT continue ; NEW_LINE DEDENT dp [ i ] [ j ] = ( dp [ i - 1 ] [ j ] + dp [ i ] [ j - 1 ] + dp [ i - 1 ] [ j - 1 ] ) ; NEW_LINE
if ( mappedMatrix [ i ] [ j ] == 0 ) : NEW_LINE INDENT dp [ i ] [ j ] = 0 ; NEW_LINE DEDENT print ( dp [ n ] [ m ] ) NEW_LINE
def preprocessMatrix ( a , n , m ) : NEW_LINE INDENT global prime NEW_LINE prime = [ 0 for i in range ( MAX ) ] NEW_LINE DEDENT
sieve ( ) ; NEW_LINE for i in range ( n ) : NEW_LINE INDENT for j in range ( m ) : NEW_LINE DEDENT
if ( prime [ a [ i ] [ j ] ] == 0 ) : NEW_LINE INDENT mappedMatrix [ i + 1 ] [ j + 1 ] = 1 ; NEW_LINE DEDENT
else : NEW_LINE INDENT mappedMatrix [ i + 1 ] [ j + 1 ] = 0 ; NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 3 ; NEW_LINE m = 3 ; NEW_LINE a = [ [ 2 , 3 , 7 ] , [ 5 , 4 , 2 ] , [ 3 , 7 , 11 ] ] ; NEW_LINE mappedMatrix = [ [ 0 for j in range ( MAX ) ] for i in range ( MAX ) ] NEW_LINE preprocessMatrix ( a , n , m ) ; NEW_LINE countPrimePath ( n , m ) ; NEW_LINE lexicographicalPath ( n , m ) ; NEW_LINE DEDENT
def isSubsetSum ( arr , n , sum ) : NEW_LINE
subset = [ [ False for x in range ( n + 1 ) ] for y in range ( sum + 1 ) ] NEW_LINE count = [ [ 0 for x in range ( n + 1 ) ] for y in range ( sum + 1 ) ] NEW_LINE
for i in range ( n + 1 ) : NEW_LINE INDENT subset [ 0 ] [ i ] = True NEW_LINE count [ 0 ] [ i ] = 0 NEW_LINE DEDENT
for i in range ( 1 , sum + 1 ) : NEW_LINE INDENT subset [ i ] [ 0 ] = False NEW_LINE count [ i ] [ 0 ] = - 1 NEW_LINE DEDENT
for i in range ( 1 , sum + 1 ) : NEW_LINE INDENT for j in range ( 1 , n + 1 ) : NEW_LINE INDENT subset [ i ] [ j ] = subset [ i ] [ j - 1 ] NEW_LINE count [ i ] [ j ] = count [ i ] [ j - 1 ] NEW_LINE if ( i >= arr [ j - 1 ] ) : NEW_LINE INDENT subset [ i ] [ j ] = ( subset [ i ] [ j ] or subset [ i - arr [ j - 1 ] ] [ j - 1 ] ) NEW_LINE if ( subset [ i ] [ j ] ) : NEW_LINE INDENT count [ i ] [ j ] = ( max ( count [ i ] [ j - 1 ] , count [ i - arr [ j - 1 ] ] [ j - 1 ] + 1 ) ) NEW_LINE DEDENT DEDENT DEDENT DEDENT return count [ sum ] [ n ] NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 2 , 3 , 5 , 10 ] NEW_LINE sum = 20 NEW_LINE n = 4 NEW_LINE print ( isSubsetSum ( arr , n , sum ) ) NEW_LINE DEDENT
MAX = 100 NEW_LINE lcslen = 0 NEW_LINE
dp = [ [ - 1 for i in range ( MAX ) ] for i in range ( MAX ) ] NEW_LINE
def lcs ( str1 , str2 , len1 , len2 , i , j ) : NEW_LINE
if ( i == len1 or j == len2 ) : NEW_LINE INDENT dp [ i ] [ j ] = 0 NEW_LINE return dp [ i ] [ j ] NEW_LINE DEDENT
if ( dp [ i ] [ j ] != - 1 ) : NEW_LINE INDENT return dp [ i ] [ j ] NEW_LINE DEDENT ret = 0 NEW_LINE
if ( str1 [ i ] == str2 [ j ] ) : NEW_LINE INDENT ret = 1 + lcs ( str1 , str2 , len1 , len2 , i + 1 , j + 1 ) NEW_LINE DEDENT else : NEW_LINE INDENT ret = max ( lcs ( str1 , str2 , len1 , len2 , i + 1 , j ) , lcs ( str1 , str2 , len1 , len2 , i , j + 1 ) ) NEW_LINE DEDENT dp [ i ] [ j ] = ret NEW_LINE return ret NEW_LINE
def printAll ( str1 , str2 , len1 , len2 , data , indx1 , indx2 , currlcs ) : NEW_LINE
if ( currlcs == lcslen ) : NEW_LINE INDENT print ( " " . join ( data [ : currlcs ] ) ) NEW_LINE return NEW_LINE DEDENT
if ( indx1 == len1 or indx2 == len2 ) : NEW_LINE INDENT return NEW_LINE DEDENT
for ch in range ( ord ( ' a ' ) , ord ( ' z ' ) + 1 ) : NEW_LINE
done = False NEW_LINE for i in range ( indx1 , len1 ) : NEW_LINE
if ( chr ( ch ) == str1 [ i ] ) : NEW_LINE for j in range ( indx2 , len2 ) : NEW_LINE
if ( chr ( ch ) == str2 [ j ] and dp [ i ] [ j ] == lcslen - currlcs ) : NEW_LINE data [ currlcs ] = chr ( ch ) NEW_LINE printAll ( str1 , str2 , len1 , len2 , data , i + 1 , j + 1 , currlcs + 1 ) NEW_LINE done = True NEW_LINE break NEW_LINE
if ( done ) : NEW_LINE INDENT break NEW_LINE DEDENT
def prinlAllLCSSorted ( str1 , str2 ) : NEW_LINE INDENT global lcslen NEW_LINE DEDENT
len1 , len2 = len ( str1 ) , len ( str2 ) NEW_LINE
lcslen = lcs ( str1 , str2 , len1 , len2 , 0 , 0 ) NEW_LINE
data = [ ' a ' for i in range ( MAX ) ] NEW_LINE printAll ( str1 , str2 , len1 , len2 , data , 0 , 0 , 0 ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT str1 = " abcabcaa " NEW_LINE str2 = " acbacba " NEW_LINE prinlAllLCSSorted ( str1 , str2 ) NEW_LINE DEDENT
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
def isPossible ( target ) : NEW_LINE
INDENT max = 0 NEW_LINE DEDENT
INDENT index = 0 NEW_LINE DEDENT
INDENT for i in range ( len ( target ) ) : NEW_LINE DEDENT
if ( max < target [ i ] ) : NEW_LINE max = target [ i ] NEW_LINE index = i NEW_LINE
INDENT if ( max == 1 ) : NEW_LINE INDENT return True NEW_LINE DEDENT DEDENT
INDENT for i in range ( len ( target ) ) : NEW_LINE DEDENT
if ( i != index ) : NEW_LINE
max -= target [ i ] NEW_LINE
if ( max <= 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT
INDENT target [ index ] = max NEW_LINE DEDENT
INDENT return isPossible ( target ) NEW_LINE DEDENT
target = [ 9 , 3 , 5 ] NEW_LINE res = isPossible ( target ) NEW_LINE if ( res ) : NEW_LINE INDENT print ( " YES " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NO " ) NEW_LINE DEDENT
def nCr ( n , r ) : NEW_LINE
res = 1 NEW_LINE
if ( r > n - r ) : NEW_LINE INDENT r = n - r NEW_LINE DEDENT
for i in range ( r ) : NEW_LINE INDENT res *= ( n - i ) NEW_LINE res //= ( i + 1 ) NEW_LINE DEDENT return res NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 3 NEW_LINE m = 2 NEW_LINE k = 2 NEW_LINE print ( nCr ( n + m , k ) ) NEW_LINE DEDENT
import math NEW_LINE
def Is_possible ( N ) : NEW_LINE INDENT C = 0 NEW_LINE D = 0 NEW_LINE DEDENT
while ( N % 10 == 0 ) : NEW_LINE INDENT N = N / 10 NEW_LINE C += 1 NEW_LINE DEDENT
if ( math . log ( N , 2 ) - int ( math . log ( N , 2 ) ) == 0 ) : NEW_LINE INDENT D = int ( math . log ( N , 2 ) ) NEW_LINE DEDENT
if ( C >= D ) : NEW_LINE INDENT print ( " YES " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NO " ) NEW_LINE DEDENT else : NEW_LINE print ( " NO " ) NEW_LINE
N = 2000000000000 NEW_LINE Is_possible ( N ) NEW_LINE
def findNthTerm ( n ) : NEW_LINE INDENT print ( n * n - n + 1 ) NEW_LINE DEDENT
N = 4 NEW_LINE findNthTerm ( N ) NEW_LINE
def rev ( num ) : NEW_LINE INDENT rev_num = 0 NEW_LINE while ( num > 0 ) : NEW_LINE INDENT rev_num = rev_num * 10 + num % 10 NEW_LINE num = num // 10 NEW_LINE DEDENT DEDENT
return rev_num NEW_LINE
def divSum ( num ) : NEW_LINE
result = 0 NEW_LINE
for i in range ( 2 , int ( num ** 0.5 ) ) : NEW_LINE
' NEW_LINE INDENT if ( num % i == 0 ) : NEW_LINE DEDENT
if ( i == ( num / i ) ) : NEW_LINE INDENT result += rev ( i ) NEW_LINE DEDENT else : NEW_LINE INDENT result += ( rev ( i ) + rev ( num / i ) ) NEW_LINE DEDENT
return ( result + 1 ) NEW_LINE
def isAntiPerfect ( n ) : NEW_LINE INDENT return divSum ( n ) == n NEW_LINE DEDENT
N = 244 NEW_LINE
if ( isAntiPerfect ( N ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
def printSeries ( n , a , b , c ) : NEW_LINE
if ( n == 1 ) : NEW_LINE INDENT print ( a , end = " ▁ " ) ; NEW_LINE return ; NEW_LINE DEDENT if ( n == 2 ) : NEW_LINE INDENT print ( a , b , end = " ▁ " ) ; NEW_LINE return ; NEW_LINE DEDENT print ( a , b , c , end = " ▁ " ) ; NEW_LINE for i in range ( 4 , n + 1 ) : NEW_LINE INDENT d = a + b + c ; NEW_LINE print ( d , end = " ▁ " ) ; NEW_LINE a = b ; NEW_LINE b = c ; NEW_LINE c = d ; NEW_LINE DEDENT
N = 7 ; a = 1 ; b = 3 ; NEW_LINE c = 4 ; NEW_LINE
printSeries ( N , a , b , c ) ; NEW_LINE
def diameter ( n ) : NEW_LINE
L , H , templen = 0 , 0 , 0 ; NEW_LINE L = 1 ; NEW_LINE
H = 0 ; NEW_LINE
if ( n == 1 ) : NEW_LINE INDENT return 1 ; NEW_LINE DEDENT if ( n == 2 ) : NEW_LINE INDENT return 2 ; NEW_LINE DEDENT if ( n == 3 ) : NEW_LINE INDENT return 3 ; NEW_LINE DEDENT
while ( L * 2 <= n ) : NEW_LINE INDENT L *= 2 ; NEW_LINE H += 1 ; NEW_LINE DEDENT
if ( n >= L * 2 - 1 ) : NEW_LINE INDENT return 2 * H + 1 ; NEW_LINE DEDENT elif ( n >= L + ( L / 2 ) - 1 ) : NEW_LINE INDENT return 2 * H ; NEW_LINE DEDENT return 2 * H - 1 ; NEW_LINE
n = 15 ; NEW_LINE print ( diameter ( n ) ) ; NEW_LINE
import math NEW_LINE
def compareValues ( a , b , c , d ) : NEW_LINE
log1 = math . log10 ( a ) NEW_LINE num1 = log1 * b NEW_LINE
log2 = math . log10 ( c ) NEW_LINE num2 = log2 * d NEW_LINE
if num1 > num2 : NEW_LINE INDENT print ( a , ' ^ ' , b ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( c , ' ^ ' , d ) NEW_LINE DEDENT
a = 8 NEW_LINE b = 29 NEW_LINE c = 60 NEW_LINE d = 59 NEW_LINE compareValues ( a , b , c , d ) NEW_LINE
MAX = 100005 NEW_LINE
def addPrimes ( ) : NEW_LINE INDENT n = MAX NEW_LINE prime = [ True for i in range ( n + 1 ) ] NEW_LINE for p in range ( 2 , n + 1 ) : NEW_LINE INDENT if p * p > n : NEW_LINE INDENT break NEW_LINE DEDENT if ( prime [ p ] == True ) : NEW_LINE INDENT for i in range ( 2 * p , n + 1 , p ) : NEW_LINE INDENT prime [ i ] = False NEW_LINE DEDENT DEDENT DEDENT ans = [ ] NEW_LINE DEDENT
for p in range ( 2 , n + 1 ) : NEW_LINE INDENT if ( prime [ p ] ) : NEW_LINE INDENT ans . append ( p ) NEW_LINE DEDENT DEDENT return ans NEW_LINE
def is_prime ( n ) : NEW_LINE INDENT if n in [ 3 , 5 , 7 ] : NEW_LINE INDENT return True NEW_LINE DEDENT return False NEW_LINE DEDENT
def find_Sum ( n ) : NEW_LINE
Sum = 0 NEW_LINE
v = addPrimes ( ) NEW_LINE
for i in range ( len ( v ) ) : NEW_LINE
flag = 1 NEW_LINE a = v [ i ] NEW_LINE
while ( a != 0 ) : NEW_LINE INDENT d = a % 10 ; NEW_LINE a = a // 10 ; NEW_LINE if ( is_prime ( d ) ) : NEW_LINE INDENT flag = 0 NEW_LINE break NEW_LINE DEDENT DEDENT
if ( flag == 1 ) : NEW_LINE INDENT n -= 1 NEW_LINE Sum = Sum + v [ i ] NEW_LINE DEDENT if n == 0 : NEW_LINE INDENT break NEW_LINE DEDENT
return Sum NEW_LINE
n = 7 NEW_LINE
print ( find_Sum ( n ) ) NEW_LINE
def primeCount ( arr , n ) : NEW_LINE
max_val = max ( arr ) NEW_LINE
prime = [ True ] * ( max_val + 1 ) NEW_LINE
prime [ 0 ] = prime [ 1 ] = False NEW_LINE p = 2 NEW_LINE while p * p <= max_val : NEW_LINE
if prime [ p ] == True : NEW_LINE
for i in range ( p * 2 , max_val + 1 , p ) : NEW_LINE INDENT prime [ i ] = False NEW_LINE DEDENT p += 1 NEW_LINE
count = 0 NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT if prime [ arr [ i ] ] : NEW_LINE INDENT count += 1 NEW_LINE DEDENT DEDENT return count NEW_LINE
def getPrefixArray ( arr , n , pre ) : NEW_LINE
pre [ 0 ] = arr [ 0 ] NEW_LINE for i in range ( 1 , n ) : NEW_LINE INDENT pre [ i ] = pre [ i - 1 ] + arr [ i ] NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 1 , 4 , 8 , 4 ] NEW_LINE n = len ( arr ) NEW_LINE DEDENT
pre = [ None ] * n NEW_LINE getPrefixArray ( arr , n , pre ) NEW_LINE
print ( primeCount ( pre , n ) ) NEW_LINE
import math NEW_LINE
def minValue ( n , x , y ) : NEW_LINE
val = ( y * n ) / 100 NEW_LINE
if x >= val : NEW_LINE INDENT return 0 NEW_LINE DEDENT else : NEW_LINE INDENT return math . ceil ( val ) - x NEW_LINE DEDENT
n = 10 ; x = 2 ; y = 40 NEW_LINE print ( minValue ( n , x , y ) ) NEW_LINE
from math import sqrt NEW_LINE
def isPrime ( n ) : NEW_LINE
if ( n <= 1 ) : NEW_LINE INDENT return False NEW_LINE DEDENT if ( n <= 3 ) : NEW_LINE INDENT return True NEW_LINE DEDENT
if ( n % 2 == 0 or n % 3 == 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT for i in range ( 5 , int ( sqrt ( n ) ) + 1 , 6 ) : NEW_LINE INDENT if ( n % i == 0 or n % ( i + 2 ) == 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT return True NEW_LINE
def isFactorialPrime ( n ) : NEW_LINE
if ( not isPrime ( n ) ) : NEW_LINE INDENT return False NEW_LINE DEDENT fact = 1 NEW_LINE i = 1 NEW_LINE while ( fact <= n + 1 ) : NEW_LINE
fact = fact * i NEW_LINE
if ( n + 1 == fact or n - 1 == fact ) : NEW_LINE INDENT return True NEW_LINE DEDENT i += 1 NEW_LINE
return False NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 23 NEW_LINE if ( isFactorialPrime ( n ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT DEDENT
n = 5 NEW_LINE
fac1 = 1 NEW_LINE for i in range ( 2 , n ) : NEW_LINE INDENT fac1 = fac1 * i NEW_LINE DEDENT
fac2 = fac1 * n NEW_LINE
totalWays = fac1 * fac2 NEW_LINE
print ( totalWays ) NEW_LINE
MAX = 10000 NEW_LINE arr = [ ] NEW_LINE
def SieveOfEratosthenes ( ) : NEW_LINE
prime = [ True ] * MAX NEW_LINE p = 2 NEW_LINE while p * p < MAX : NEW_LINE
if ( prime [ p ] == True ) : NEW_LINE
for i in range ( p * 2 , MAX , p ) : NEW_LINE INDENT prime [ i ] = False NEW_LINE DEDENT p += 1 NEW_LINE
' NEW_LINE INDENT for p in range ( 2 , MAX ) : NEW_LINE INDENT if ( prime [ p ] ) : NEW_LINE INDENT arr . append ( p ) NEW_LINE DEDENT DEDENT DEDENT
def isEuclid ( n ) : NEW_LINE INDENT product = 1 NEW_LINE i = 0 NEW_LINE while ( product < n ) : NEW_LINE DEDENT
product = product * arr [ i ] NEW_LINE if ( product + 1 == n ) : NEW_LINE INDENT return True NEW_LINE DEDENT i += 1 NEW_LINE return False NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
SieveOfEratosthenes ( ) NEW_LINE
n = 31 NEW_LINE
if ( isEuclid ( n ) ) : NEW_LINE INDENT print ( " YES " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NO " ) NEW_LINE DEDENT
n = 42 NEW_LINE
if ( isEuclid ( n ) ) : NEW_LINE INDENT print ( " YES " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NO " ) NEW_LINE DEDENT
from math import * NEW_LINE
def nextPerfectCube ( N ) : NEW_LINE INDENT nextN = floor ( N ** ( 1 / 3 ) ) + 1 NEW_LINE return nextN ** 3 NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 35 NEW_LINE print ( nextPerfectCube ( n ) ) NEW_LINE DEDENT
import math NEW_LINE
def isPrime ( n ) : NEW_LINE
if ( n <= 1 ) : NEW_LINE INDENT return False NEW_LINE DEDENT if ( n <= 3 ) : NEW_LINE INDENT return True NEW_LINE DEDENT
if ( n % 2 == 0 or n % 3 == 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT i = 5 NEW_LINE while i * i <= n : NEW_LINE INDENT if ( n % i == 0 or n % ( i + 2 ) == 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT i = i + 6 NEW_LINE DEDENT return True NEW_LINE
def SumOfPrimeDivisors ( n ) : NEW_LINE INDENT Sum = 0 NEW_LINE DEDENT
root_n = ( int ) ( math . sqrt ( n ) ) NEW_LINE for i in range ( 1 , root_n + 1 ) : NEW_LINE INDENT if ( n % i == 0 ) : NEW_LINE DEDENT
if ( i == ( int ) ( n / i ) and isPrime ( i ) ) : NEW_LINE INDENT Sum += i NEW_LINE DEDENT else : NEW_LINE
if ( isPrime ( i ) ) : NEW_LINE INDENT Sum += i NEW_LINE DEDENT if ( isPrime ( ( int ) ( n / i ) ) ) : NEW_LINE INDENT Sum += ( int ) ( n / i ) NEW_LINE DEDENT return Sum NEW_LINE
n = 60 NEW_LINE print ( " Sum ▁ of ▁ prime ▁ divisors ▁ of ▁ 60 ▁ is " , SumOfPrimeDivisors ( n ) ) NEW_LINE
def findpos ( n ) : NEW_LINE INDENT pos = 0 NEW_LINE for i in n : NEW_LINE DEDENT
if i == '2' : NEW_LINE INDENT pos = pos * 4 + 1 NEW_LINE DEDENT
elif i == '3' : NEW_LINE INDENT pos = pos * 4 + 2 NEW_LINE DEDENT
elif i == '5' : NEW_LINE INDENT pos = pos * 4 + 3 NEW_LINE DEDENT
elif i == '7' : NEW_LINE INDENT pos = pos * 4 + 4 NEW_LINE DEDENT return pos NEW_LINE
n = "777" NEW_LINE print ( findpos ( n ) ) NEW_LINE
def possibleTripletInRange ( L , R ) : NEW_LINE INDENT flag = False ; NEW_LINE possibleA = 0 ; NEW_LINE possibleB = 0 ; NEW_LINE possibleC = 0 ; NEW_LINE numbersInRange = ( R - L + 1 ) ; NEW_LINE DEDENT
if ( numbersInRange < 3 ) : NEW_LINE INDENT flag = False ; NEW_LINE DEDENT
elif ( numbersInRange > 3 ) : NEW_LINE INDENT flag = True ; NEW_LINE DEDENT
if ( ( L % 2 ) > 0 ) : NEW_LINE INDENT L += 1 ; NEW_LINE DEDENT possibleA = L ; NEW_LINE possibleB = L + 1 ; NEW_LINE possibleC = L + 2 ; NEW_LINE else : NEW_LINE
if ( ( L % 2 ) == 0 ) : NEW_LINE INDENT flag = True ; NEW_LINE possibleA = L ; NEW_LINE possibleB = L + 1 ; NEW_LINE possibleC = L + 2 ; NEW_LINE DEDENT else : NEW_LINE
flag = False ; NEW_LINE
if ( flag == True ) : NEW_LINE INDENT print ( " ( " , possibleA , " , " , possibleB , " , " , possibleC , " ) ▁ is ▁ one ▁ such " , " possible ▁ triplet ▁ between " , L , " and " , R ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No ▁ Such ▁ Triplet ▁ exists ▁ between " , L , " and " , R ) ; NEW_LINE DEDENT
L = 2 ; NEW_LINE R = 10 ; NEW_LINE possibleTripletInRange ( L , R ) ; NEW_LINE
L = 23 ; NEW_LINE R = 46 ; NEW_LINE possibleTripletInRange ( L , R ) ; NEW_LINE
mod = 1000000007 NEW_LINE
def digitNumber ( n ) : NEW_LINE
if ( n == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT
if ( n == 1 ) : NEW_LINE INDENT return 9 NEW_LINE DEDENT
if ( n % 2 != 0 ) : NEW_LINE
temp = digitNumber ( ( n - 1 ) // 2 ) % mod NEW_LINE return ( 9 * ( temp * temp ) % mod ) % mod NEW_LINE else : NEW_LINE
temp = digitNumber ( n // 2 ) % mod NEW_LINE return ( temp * temp ) % mod NEW_LINE def countExcluding ( n , d ) : NEW_LINE
if ( d == 0 ) : NEW_LINE INDENT return ( 9 * digitNumber ( n - 1 ) ) % mod NEW_LINE DEDENT else : NEW_LINE INDENT return ( 8 * digitNumber ( n - 1 ) ) % mod NEW_LINE DEDENT
d = 9 NEW_LINE n = 3 NEW_LINE print ( countExcluding ( n , d ) ) NEW_LINE
def isPrime ( n ) : NEW_LINE
if n <= 1 : NEW_LINE INDENT return False NEW_LINE DEDENT
for i in range ( 2 , n ) : NEW_LINE INDENT if n % i == 0 : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT return True NEW_LINE
def isEmirp ( n ) : NEW_LINE
n = int ( n ) NEW_LINE if isPrime ( n ) == False : NEW_LINE INDENT return False NEW_LINE DEDENT
rev = 0 NEW_LINE while n != 0 : NEW_LINE INDENT d = n % 10 NEW_LINE rev = rev * 10 + d NEW_LINE n = int ( n / 10 ) NEW_LINE DEDENT
return isPrime ( rev ) NEW_LINE
n = 13 NEW_LINE if isEmirp ( n ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
def Convert ( radian ) : NEW_LINE INDENT pi = 3.14159 NEW_LINE degree = radian * ( 180 / pi ) NEW_LINE return degree NEW_LINE DEDENT
radian = 5 NEW_LINE print ( " degree ▁ = " , ( Convert ( radian ) ) ) NEW_LINE
def sn ( n , an ) : NEW_LINE INDENT return ( n * ( 1 + an ) ) / 2 ; NEW_LINE DEDENT
def trace ( n , m ) : NEW_LINE
an = 1 + ( n - 1 ) * ( m + 1 ) ; NEW_LINE
rowmajorSum = sn ( n , an ) ; NEW_LINE
an = 1 + ( n - 1 ) * ( n + 1 ) ; NEW_LINE
colmajorSum = sn ( n , an ) ; NEW_LINE return int ( rowmajorSum + colmajorSum ) ; NEW_LINE
N = 3 ; NEW_LINE M = 3 ; NEW_LINE print ( trace ( N , M ) ) ; NEW_LINE
def max_area ( n , m , k ) : NEW_LINE INDENT if ( k > ( n + m - 2 ) ) : NEW_LINE INDENT print ( " Not ▁ possible " ) NEW_LINE DEDENT else : NEW_LINE DEDENT
if ( k < max ( m , n ) - 1 ) : NEW_LINE INDENT result = max ( m * ( n / ( k + 1 ) ) , n * ( m / ( k + 1 ) ) ) ; NEW_LINE DEDENT
else : NEW_LINE INDENT result = max ( m / ( k - n + 2 ) , n / ( k - m + 2 ) ) ; NEW_LINE DEDENT
print ( result ) NEW_LINE
n = 3 NEW_LINE m = 4 NEW_LINE k = 1 NEW_LINE max_area ( n , m , k ) NEW_LINE
def area_fun ( side ) : NEW_LINE INDENT area = side * side NEW_LINE return area NEW_LINE DEDENT
side = 4 NEW_LINE area = area_fun ( side ) NEW_LINE print ( area ) NEW_LINE
def countConsecutive ( N ) : NEW_LINE
count = 0 NEW_LINE L = 1 NEW_LINE while ( L * ( L + 1 ) < 2 * N ) : NEW_LINE INDENT a = ( 1.0 * N - ( L * ( L + 1 ) ) / 2 ) / ( L + 1 ) NEW_LINE if ( a - int ( a ) == 0.0 ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT L += 1 NEW_LINE DEDENT return count NEW_LINE
N = 15 NEW_LINE print countConsecutive ( N ) NEW_LINE N = 10 NEW_LINE print countConsecutive ( N ) NEW_LINE
def isAutomorphic ( N ) : NEW_LINE
sq = N * N NEW_LINE
while ( N > 0 ) : NEW_LINE
if ( N % 10 != sq % 10 ) : NEW_LINE INDENT return False NEW_LINE DEDENT
N /= 10 NEW_LINE sq /= 10 NEW_LINE return True NEW_LINE
N = 5 NEW_LINE if isAutomorphic ( N ) : NEW_LINE INDENT print " Automorphic " NEW_LINE DEDENT else : NEW_LINE INDENT print " Not ▁ Automorphic " NEW_LINE DEDENT
def maxPrimefactorNum ( N ) : NEW_LINE
arr = [ True ] * ( N + 5 ) ; NEW_LINE
i = 3 ; NEW_LINE while ( i * i <= N ) : NEW_LINE INDENT if ( arr [ i ] ) : NEW_LINE INDENT for j in range ( i * i , N + 1 , i ) : NEW_LINE INDENT arr [ j ] = False ; NEW_LINE DEDENT DEDENT i += 2 ; NEW_LINE DEDENT
prime = [ ] ; NEW_LINE prime . append ( 2 ) ; NEW_LINE for i in range ( 3 , N + 1 , 2 ) : NEW_LINE INDENT if ( arr [ i ] ) : NEW_LINE INDENT prime . append ( i ) ; NEW_LINE DEDENT DEDENT
i = 0 ; NEW_LINE ans = 1 ; NEW_LINE while ( ans * prime [ i ] <= N and i < len ( prime ) ) : NEW_LINE INDENT ans *= prime [ i ] ; NEW_LINE i += 1 ; NEW_LINE DEDENT return ans ; NEW_LINE
N = 40 ; NEW_LINE print ( maxPrimefactorNum ( N ) ) ; NEW_LINE
import math NEW_LINE
def divSum ( num ) : NEW_LINE
result = 0 NEW_LINE
' NEW_LINE INDENT i = 2 NEW_LINE while i <= ( math . sqrt ( num ) ) : NEW_LINE DEDENT
' NEW_LINE INDENT if ( num % i == 0 ) : NEW_LINE DEDENT
if ( i == ( num / i ) ) : NEW_LINE INDENT result = result + i ; NEW_LINE DEDENT else : NEW_LINE INDENT result = result + ( i + num / i ) ; NEW_LINE DEDENT i = i + 1 NEW_LINE
return ( result + 1 ) ; NEW_LINE
num = 36 NEW_LINE print ( divSum ( num ) ) NEW_LINE
def power ( x , y , p ) : NEW_LINE
while ( y > 0 ) : NEW_LINE
if ( y & 1 ) : NEW_LINE INDENT res = ( res * x ) % p NEW_LINE DEDENT
x = ( x * x ) % p NEW_LINE return res NEW_LINE
def squareRoot ( n , p ) : NEW_LINE INDENT if ( p % 4 != 3 ) : NEW_LINE INDENT print ( " Invalid ▁ Input " ) NEW_LINE return NEW_LINE DEDENT DEDENT
n = n % p NEW_LINE x = power ( n , ( p + 1 ) // 4 , p ) NEW_LINE if ( ( x * x ) % p == n ) : NEW_LINE INDENT print ( " Square ▁ root ▁ is ▁ " , x ) NEW_LINE return NEW_LINE DEDENT
x = p - x NEW_LINE if ( ( x * x ) % p == n ) : NEW_LINE INDENT print ( " Square ▁ root ▁ is ▁ " , x ) NEW_LINE return NEW_LINE DEDENT
print ( " Square ▁ root ▁ doesn ' t ▁ exist ▁ " ) NEW_LINE
p = 7 NEW_LINE n = 2 NEW_LINE squareRoot ( n , p ) NEW_LINE
import random NEW_LINE
def power ( x , y , p ) : NEW_LINE
res = 1 ; NEW_LINE
x = x % p ; NEW_LINE while ( y > 0 ) : NEW_LINE
if ( y & 1 ) : NEW_LINE INDENT res = ( res * x ) % p ; NEW_LINE DEDENT
x = ( x * x ) % p ; NEW_LINE return res ; NEW_LINE
def miillerTest ( d , n ) : NEW_LINE
a = 2 + random . randint ( 1 , n - 4 ) ; NEW_LINE
x = power ( a , d , n ) ; NEW_LINE if ( x == 1 or x == n - 1 ) : NEW_LINE INDENT return True ; NEW_LINE DEDENT
while ( d != n - 1 ) : NEW_LINE INDENT x = ( x * x ) % n ; NEW_LINE d *= 2 ; NEW_LINE if ( x == 1 ) : NEW_LINE INDENT return False ; NEW_LINE DEDENT if ( x == n - 1 ) : NEW_LINE INDENT return True ; NEW_LINE DEDENT DEDENT
return False ; NEW_LINE
def isPrime ( n , k ) : NEW_LINE
if ( n <= 1 or n == 4 ) : NEW_LINE INDENT return False ; NEW_LINE DEDENT if ( n <= 3 ) : NEW_LINE INDENT return True ; NEW_LINE DEDENT
d = n - 1 ; NEW_LINE while ( d % 2 == 0 ) : NEW_LINE INDENT d //= 2 ; NEW_LINE DEDENT
for i in range ( k ) : NEW_LINE INDENT if ( miillerTest ( d , n ) == False ) : NEW_LINE INDENT return False ; NEW_LINE DEDENT DEDENT return True ; NEW_LINE
k = 4 ; NEW_LINE print ( " All ▁ primes ▁ smaller ▁ than ▁ 100 : ▁ " ) ; NEW_LINE for n in range ( 1 , 100 ) : NEW_LINE INDENT if ( isPrime ( n , k ) ) : NEW_LINE INDENT print ( n , end = " ▁ " ) ; NEW_LINE DEDENT DEDENT
def maxConsecutiveOnes ( x ) : NEW_LINE
count = 0 NEW_LINE
while ( x != 0 ) : NEW_LINE
x = ( x & ( x << 1 ) ) NEW_LINE count = count + 1 NEW_LINE return count NEW_LINE
print ( maxConsecutiveOnes ( 14 ) ) NEW_LINE print ( maxConsecutiveOnes ( 222 ) ) NEW_LINE
def subtract ( x , y ) : NEW_LINE
while ( y != 0 ) : NEW_LINE
borrow = ( ~ x ) & y NEW_LINE
x = x ^ y NEW_LINE
y = borrow << 1 NEW_LINE return x NEW_LINE
x = 29 NEW_LINE y = 13 NEW_LINE print ( " x ▁ - ▁ y ▁ is " , subtract ( x , y ) ) NEW_LINE
def subtract ( x , y ) : NEW_LINE INDENT if ( y == 0 ) : NEW_LINE INDENT return x NEW_LINE DEDENT return subtract ( x ^ y , ( ~ x & y ) << 1 ) NEW_LINE DEDENT
x = 29 NEW_LINE y = 13 NEW_LINE print ( " x ▁ - ▁ y ▁ is " , subtract ( x , y ) ) NEW_LINE
def addEdge ( v , x , y ) : NEW_LINE INDENT v [ x ] . append ( y ) NEW_LINE v [ y ] . append ( x ) NEW_LINE DEDENT
def dfs ( tree , temp , ancestor , u , parent , k ) : NEW_LINE
temp . append ( u ) NEW_LINE
for i in tree [ u ] : NEW_LINE INDENT if ( i == parent ) : NEW_LINE INDENT continue NEW_LINE DEDENT dfs ( tree , temp , ancestor , i , u , k ) NEW_LINE DEDENT temp . pop ( ) NEW_LINE
if ( len ( temp ) < k ) : NEW_LINE INDENT ancestor [ u ] = - 1 NEW_LINE DEDENT else : NEW_LINE
ancestor [ u ] = temp [ len ( temp ) - k ] NEW_LINE
def KthAncestor ( N , K , E , edges ) : NEW_LINE
tree = [ [ ] for i in range ( N + 1 ) ] NEW_LINE for i in range ( E ) : NEW_LINE INDENT addEdge ( tree , edges [ i ] [ 0 ] , edges [ i ] [ 1 ] ) NEW_LINE DEDENT
temp = [ ] NEW_LINE
ancestor = [ 0 ] * ( N + 1 ) NEW_LINE dfs ( tree , temp , ancestor , 1 , 0 , K ) NEW_LINE
for i in range ( 1 , N + 1 ) : NEW_LINE INDENT print ( ancestor [ i ] , end = " ▁ " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE
N = 9 NEW_LINE K = 2 NEW_LINE
E = 8 NEW_LINE edges = [ [ 1 , 2 ] , [ 1 , 3 ] , [ 2 , 4 ] , [ 2 , 5 ] , [ 2 , 6 ] , [ 3 , 7 ] , [ 3 , 8 ] , [ 3 , 9 ] ] NEW_LINE
KthAncestor ( N , K , E , edges ) NEW_LINE
def build ( sum , a , l , r , rt ) : NEW_LINE
if ( l == r ) : NEW_LINE INDENT sum [ rt ] = a [ l - 1 ] NEW_LINE return NEW_LINE DEDENT
m = ( l + r ) >> 1 NEW_LINE
build ( sum , a , l , m , rt << 1 ) NEW_LINE build ( sum , a , m + 1 , r , rt << 1 1 ) NEW_LINE
def pushDown ( sum , add , rt , ln , rn ) : NEW_LINE INDENT if ( add [ rt ] ) : NEW_LINE INDENT add [ rt << 1 ] += add [ rt ] NEW_LINE add [ rt << 1 1 ] += add [ rt ] NEW_LINE sum [ rt << 1 ] += add [ rt ] * ln NEW_LINE sum [ rt << 1 1 ] += add [ rt ] * rn NEW_LINE add [ rt ] = 0 NEW_LINE DEDENT DEDENT
def update ( sum , add , L , R , C , l , r , rt ) : NEW_LINE
if ( L <= l and r <= R ) : NEW_LINE INDENT sum [ rt ] += C * ( r - l + 1 ) NEW_LINE add [ rt ] += C NEW_LINE return NEW_LINE DEDENT
m = ( l + r ) >> 1 NEW_LINE
pushDown ( sum , add , rt , m - l + 1 , r - m ) NEW_LINE
if ( L <= m ) : NEW_LINE INDENT update ( sum , add , L , R , C , l , m , rt << 1 ) NEW_LINE DEDENT if ( R > m ) : NEW_LINE INDENT update ( sum , add , L , R , C , m + 1 , r , rt << 1 1 ) NEW_LINE DEDENT
def queryy ( sum , add , L , R , l , r , rt ) : NEW_LINE
if ( L <= l and r <= R ) : NEW_LINE INDENT return sum [ rt ] NEW_LINE DEDENT
m = ( l + r ) >> 1 NEW_LINE
pushDown ( sum , add , rt , m - l + 1 , r - m ) NEW_LINE ans = 0 NEW_LINE
if ( L <= m ) : NEW_LINE INDENT ans += queryy ( sum , add , L , R , l , m , rt << 1 ) NEW_LINE DEDENT if ( R > m ) : NEW_LINE INDENT ans += queryy ( sum , add , L , R , m + 1 , r , ( rt << 1 1 ) ) NEW_LINE DEDENT
return ans NEW_LINE
def sequenceMaintenance ( n , q , a , b , m ) : NEW_LINE
a = sorted ( a ) NEW_LINE
sum = [ 0 ] * ( 4 * n ) NEW_LINE add = [ 0 ] * ( 4 * n ) NEW_LINE ans = [ ] NEW_LINE
build ( sum , a , 1 , n , 1 ) NEW_LINE
for i in range ( q ) : NEW_LINE INDENT l = 1 NEW_LINE r = n NEW_LINE pos = - 1 NEW_LINE while ( l <= r ) : NEW_LINE INDENT m = ( l + r ) >> 1 NEW_LINE if ( queryy ( sum , add , m , m , 1 , n , 1 ) >= b [ i ] ) : NEW_LINE INDENT r = m - 1 NEW_LINE pos = m NEW_LINE DEDENT else : NEW_LINE INDENT l = m + 1 NEW_LINE DEDENT DEDENT if ( pos == - 1 ) : NEW_LINE INDENT ans . append ( 0 ) NEW_LINE DEDENT else : NEW_LINE DEDENT
ans . append ( n - pos + 1 ) NEW_LINE
update ( sum , add , pos , n , - m , 1 , n , 1 ) NEW_LINE
for i in ans : NEW_LINE INDENT print ( i , end = " ▁ " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 4 NEW_LINE Q = 3 NEW_LINE M = 1 NEW_LINE arr = [ 1 , 2 , 3 , 4 ] NEW_LINE query = [ 4 , 3 , 1 ] NEW_LINE DEDENT
sequenceMaintenance ( N , Q , arr , query , M ) NEW_LINE
import math NEW_LINE
def hasCoprimePair ( arr , n ) : NEW_LINE
for i in range ( n - 1 ) : NEW_LINE INDENT for j in range ( i + 1 , n ) : NEW_LINE DEDENT
if ( math . gcd ( arr [ i ] , arr [ j ] ) == 1 ) : NEW_LINE INDENT return True NEW_LINE DEDENT
return False NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 3 NEW_LINE arr = [ 6 , 9 , 15 ] NEW_LINE DEDENT
if ( hasCoprimePair ( arr , n ) ) : NEW_LINE INDENT print ( 1 ) NEW_LINE DEDENT
else : NEW_LINE INDENT print ( n ) NEW_LINE DEDENT
def Numberofways ( n ) : NEW_LINE INDENT count = 0 NEW_LINE for a in range ( 1 , n ) : NEW_LINE INDENT for b in range ( 1 , n ) : NEW_LINE INDENT c = n - ( a + b ) NEW_LINE DEDENT DEDENT DEDENT
if ( a < b + c and b < a + c and c < a + b ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT
return count ; NEW_LINE
n = 15 NEW_LINE print ( Numberofways ( n ) ) NEW_LINE
def countPairs ( N , arr ) : NEW_LINE INDENT count = 0 NEW_LINE DEDENT
for i in range ( N ) : NEW_LINE INDENT if ( i == arr [ arr [ i ] - 1 ] - 1 ) : NEW_LINE DEDENT
count += 1 NEW_LINE
print ( count // 2 ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 2 , 1 , 4 , 3 ] NEW_LINE N = len ( arr ) NEW_LINE countPairs ( N , arr ) NEW_LINE DEDENT
def LongestFibSubseq ( A , n ) : NEW_LINE
S = set ( A ) NEW_LINE maxLen = 0 NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT for j in range ( i + 1 , n ) : NEW_LINE INDENT x = A [ j ] NEW_LINE y = A [ i ] + A [ j ] NEW_LINE length = 2 NEW_LINE DEDENT DEDENT
while y in S : NEW_LINE
z = x + y NEW_LINE x = y NEW_LINE y = z NEW_LINE length += 1 NEW_LINE maxLen = max ( maxLen , length ) NEW_LINE return maxLen if maxLen >= 3 else 0 NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT A = [ 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 ] NEW_LINE n = len ( A ) NEW_LINE print ( LongestFibSubseq ( A , n ) ) NEW_LINE DEDENT
def CountMaximum ( arr , n , k ) : NEW_LINE
arr . sort ( ) NEW_LINE Sum , count = 0 , 0 NEW_LINE
for i in range ( 0 , n ) : NEW_LINE
Sum += arr [ i ] NEW_LINE
if ( Sum > k ) : NEW_LINE INDENT break NEW_LINE DEDENT
count += 1 NEW_LINE
return count NEW_LINE
arr = [ 30 , 30 , 10 , 10 ] NEW_LINE n = len ( arr ) NEW_LINE k = 50 NEW_LINE
print ( CountMaximum ( arr , n , k ) ) NEW_LINE
def num_candyTypes ( candies ) : NEW_LINE
s = set ( ) NEW_LINE
for i in range ( len ( candies ) ) : NEW_LINE INDENT s . add ( candies [ i ] ) NEW_LINE DEDENT
return len ( s ) NEW_LINE
def distribute_candies ( candies ) : NEW_LINE
allowed = len ( candies ) / 2 NEW_LINE
types = num_candyTypes ( candies ) NEW_LINE
if ( types < allowed ) : NEW_LINE INDENT print ( int ( types ) ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( int ( allowed ) ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE
candies = [ 4 , 4 , 5 , 5 , 3 , 3 ] NEW_LINE
distribute_candies ( candies ) NEW_LINE
import math NEW_LINE
def Length_Diagonals ( a , theta ) : NEW_LINE INDENT p = a * math . sqrt ( 2 + ( 2 * math . cos ( math . radians ( theta ) ) ) ) NEW_LINE q = a * math . sqrt ( 2 - ( 2 * math . cos ( math . radians ( theta ) ) ) ) NEW_LINE return [ p , q ] NEW_LINE DEDENT
A = 6 NEW_LINE theta = 45 NEW_LINE ans = Length_Diagonals ( A , theta ) NEW_LINE print ( round ( ans [ 0 ] , 2 ) , round ( ans [ 1 ] , 2 ) ) NEW_LINE
def countEvenOdd ( arr , n , K ) : NEW_LINE INDENT even = 0 ; odd = 0 ; NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE
x = bin ( arr [ i ] ) . count ( '1' ) ; NEW_LINE if ( x % 2 == 0 ) : NEW_LINE INDENT even += 1 ; NEW_LINE DEDENT else : NEW_LINE INDENT odd += 1 ; NEW_LINE DEDENT
y = bin ( K ) . count ( '1' ) ; NEW_LINE
if ( y & 1 ) : NEW_LINE INDENT print ( " Even ▁ = " , odd , " , ▁ Odd ▁ = " , even ) ; NEW_LINE DEDENT
else : NEW_LINE INDENT print ( " Even ▁ = " , even , " , ▁ Odd ▁ = " , odd ) ; NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 4 , 2 , 15 , 9 , 8 , 8 ] ; NEW_LINE K = 3 ; NEW_LINE n = len ( arr ) ; NEW_LINE DEDENT
countEvenOdd ( arr , n , K ) ; NEW_LINE
N = 6 NEW_LINE Even = N // 2 NEW_LINE Odd = N - Even NEW_LINE print ( Even * Odd ) NEW_LINE
import sys NEW_LINE
def longestSubSequence ( A , N , ind = 0 , lastf = - sys . maxsize - 1 , lasts = sys . maxsize ) : NEW_LINE
if ( ind == N ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
ans = longestSubSequence ( A , N , ind + 1 , lastf , lasts ) NEW_LINE
if ( A [ ind ] [ 0 ] > lastf and A [ ind ] [ 1 ] < lasts ) : NEW_LINE INDENT ans = max ( ans , longestSubSequence ( A , N , ind + 1 , A [ ind ] [ 0 ] , A [ ind ] [ 1 ] ) + 1 ) NEW_LINE DEDENT return ans NEW_LINE
/ * Function * / NEW_LINE public static int longestSubSequence ( int [ , ] A , int N ) NEW_LINE { return longestSubSequence ( A , N , 0 , 0 , 0 ) ; } NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
A = [ [ 1 , 2 ] , [ 2 , 2 ] , [ 3 , 1 ] ] NEW_LINE N = len ( A ) NEW_LINE
print ( longestSubSequence ( A , N ) ) NEW_LINE
def countTriplets ( A ) : NEW_LINE
cnt = 0 ; NEW_LINE
tuples = { } ; NEW_LINE
for a in A : NEW_LINE
for b in A : NEW_LINE INDENT if ( a & b ) in tuples : NEW_LINE INDENT tuples [ a & b ] += 1 ; NEW_LINE DEDENT else : NEW_LINE INDENT tuples [ a & b ] = 1 ; NEW_LINE DEDENT DEDENT
for a in A : NEW_LINE
for t in tuples : NEW_LINE
if ( ( t & a ) == 0 ) : NEW_LINE INDENT cnt += tuples [ t ] ; NEW_LINE DEDENT
return cnt ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
A = [ 2 , 1 , 3 ] ; NEW_LINE
print ( countTriplets ( A ) ) ; NEW_LINE
def CountWays ( n ) : NEW_LINE
noOfWays = [ 0 ] * ( n + 3 ) NEW_LINE noOfWays [ 0 ] = 1 NEW_LINE noOfWays [ 1 ] = 1 NEW_LINE noOfWays [ 2 ] = 1 + 1 NEW_LINE
for i in range ( 3 , n + 1 ) : NEW_LINE
noOfWays [ i ] = noOfWays [ i - 1 ] + noOfWays [ i - 3 ] NEW_LINE return noOfWays [ n ] NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 5 NEW_LINE print ( CountWays ( n ) ) NEW_LINE DEDENT
import sys NEW_LINE
def findWinner ( a , n ) : NEW_LINE
v = [ ] NEW_LINE
c = 0 NEW_LINE
for i in range ( 0 , n ) : NEW_LINE
if ( a [ i ] == '0' ) : NEW_LINE INDENT c += 1 NEW_LINE DEDENT
else : NEW_LINE INDENT if ( c != 0 ) : NEW_LINE INDENT v . append ( c ) NEW_LINE DEDENT c = 0 NEW_LINE DEDENT if ( c != 0 ) : NEW_LINE v . append ( c ) NEW_LINE
if ( len ( v ) == 0 ) : NEW_LINE INDENT print ( " Player ▁ B " , end = " " ) NEW_LINE return NEW_LINE DEDENT
if ( len ( v ) == 1 ) : NEW_LINE INDENT if ( ( v [ 0 ] & 1 ) != 0 ) : NEW_LINE INDENT print ( " Player ▁ A " , end = " " ) NEW_LINE DEDENT DEDENT
else : NEW_LINE INDENT print ( " Player ▁ B " , end = " " ) NEW_LINE DEDENT return NEW_LINE
first = sys . minsize NEW_LINE second = sys . minsize NEW_LINE
for i in range ( len ( v ) ) : NEW_LINE
if ( a [ i ] > first ) : NEW_LINE INDENT second = first NEW_LINE first = a [ i ] NEW_LINE DEDENT
elif ( a [ i ] > second and a [ i ] != first ) : NEW_LINE INDENT second = a [ i ] NEW_LINE DEDENT
if ( ( ( first & 1 ) != 0 ) and ( first + 1 ) // 2 > second ) : NEW_LINE INDENT print ( " Player ▁ A " , end = " " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Player ▁ B " , end = " " ) NEW_LINE DEDENT
S = "1100011" NEW_LINE N = len ( S ) NEW_LINE findWinner ( S , N ) NEW_LINE
def can_Construct ( S , K ) : NEW_LINE
m = dict ( ) NEW_LINE p = 0 NEW_LINE
if ( len ( S ) == K ) : NEW_LINE INDENT return True NEW_LINE DEDENT
for i in S : NEW_LINE INDENT m [ i ] = m . get ( i , 0 ) + 1 NEW_LINE DEDENT
if ( K > len ( S ) ) : NEW_LINE INDENT return False NEW_LINE DEDENT else : NEW_LINE
for h in m : NEW_LINE INDENT if ( m [ h ] % 2 != 0 ) : NEW_LINE INDENT p = p + 1 NEW_LINE DEDENT DEDENT
if ( K < p ) : NEW_LINE INDENT return False NEW_LINE DEDENT return True NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT S = " annabelle " NEW_LINE K = 4 NEW_LINE if ( can_Construct ( S , K ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT DEDENT
def equalIgnoreCase ( str1 , str2 ) : NEW_LINE
str1 = str1 . lower ( ) ; NEW_LINE str2 = str2 . lower ( ) ; NEW_LINE
x = str1 == str2 ; NEW_LINE
return x ; NEW_LINE
def equalIgnoreCaseUtil ( str1 , str2 ) : NEW_LINE INDENT res = equalIgnoreCase ( str1 , str2 ) ; NEW_LINE if ( res == True ) : NEW_LINE INDENT print ( " Same " ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Not ▁ Same " ) ; NEW_LINE DEDENT DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT str1 = " Geeks " ; NEW_LINE str2 = " geeks " ; NEW_LINE equalIgnoreCaseUtil ( str1 , str2 ) ; NEW_LINE str1 = " Geek " ; NEW_LINE str2 = " geeksforgeeks " ; NEW_LINE equalIgnoreCaseUtil ( str1 , str2 ) ; NEW_LINE DEDENT
import math as mt NEW_LINE
def steps ( string , n ) : NEW_LINE
flag = False NEW_LINE x = 0 NEW_LINE
for i in range ( len ( string ) ) : NEW_LINE
if ( x == 0 ) : NEW_LINE INDENT flag = True NEW_LINE DEDENT
if ( x == n - 1 ) : NEW_LINE INDENT flag = False NEW_LINE DEDENT
for j in range ( x ) : NEW_LINE INDENT print ( " * " , end = " " ) NEW_LINE DEDENT print ( string [ i ] ) NEW_LINE
if ( flag == True ) : NEW_LINE INDENT x += 1 NEW_LINE DEDENT else : NEW_LINE INDENT x -= 1 NEW_LINE DEDENT
n = 4 NEW_LINE string = " GeeksForGeeks " NEW_LINE print ( " String : ▁ " , string ) NEW_LINE print ( " Max ▁ Length ▁ of ▁ Steps : ▁ " , n ) NEW_LINE
steps ( string , n ) NEW_LINE
def countFreq ( arr , n ) : NEW_LINE
visited = [ False for i in range ( n ) ] NEW_LINE
for i in range ( n ) : NEW_LINE
if visited [ i ] == True : NEW_LINE INDENT continue NEW_LINE DEDENT
count = 1 NEW_LINE for j in range ( i + 1 , n ) : NEW_LINE INDENT if arr [ i ] == arr [ j ] : NEW_LINE INDENT visited [ j ] = True NEW_LINE count += 1 NEW_LINE DEDENT DEDENT print ( arr [ i ] , count ) NEW_LINE
a = [ 10 , 20 , 20 , 10 , 10 , 20 , 5 , 20 ] NEW_LINE n = len ( a ) NEW_LINE countFreq ( a , n ) NEW_LINE
def isDivisible ( str , k ) : NEW_LINE INDENT n = len ( str ) NEW_LINE c = 0 NEW_LINE DEDENT
for i in range ( 0 , k ) : NEW_LINE INDENT if ( str [ n - i - 1 ] == '0' ) : NEW_LINE INDENT c += 1 NEW_LINE DEDENT DEDENT
return ( c == k ) NEW_LINE
str1 = "10101100" NEW_LINE k = 2 NEW_LINE if ( isDivisible ( str1 , k ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
str2 = "111010100" NEW_LINE k = 2 NEW_LINE if ( isDivisible ( str2 , k ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
NO_OF_CHARS = 256 NEW_LINE
def canFormPalindrome ( string ) : NEW_LINE
count = [ 0 for i in range ( NO_OF_CHARS ) ] NEW_LINE
for i in string : NEW_LINE INDENT count [ ord ( i ) ] += 1 NEW_LINE DEDENT
odd = 0 NEW_LINE for i in range ( NO_OF_CHARS ) : NEW_LINE INDENT if ( count [ i ] & 1 ) : NEW_LINE INDENT odd += 1 NEW_LINE DEDENT if ( odd > 1 ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT
return True NEW_LINE
if ( canFormPalindrome ( " geeksforgeeks " ) ) : NEW_LINE INDENT print " Yes " NEW_LINE DEDENT else : NEW_LINE INDENT print " No " NEW_LINE DEDENT if ( canFormPalindrome ( " geeksogeeks " ) ) : NEW_LINE INDENT print " Yes " NEW_LINE DEDENT else : NEW_LINE INDENT print " NO " NEW_LINE DEDENT
def isNumber ( s ) : NEW_LINE INDENT for i in range ( len ( s ) ) : NEW_LINE INDENT if s [ i ] . isdigit ( ) != True : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT return True NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE
str = "6790" NEW_LINE
if isNumber ( str ) : NEW_LINE INDENT print ( " Integer " ) NEW_LINE DEDENT
else : NEW_LINE INDENT print ( " String " ) NEW_LINE DEDENT
def reverse ( string ) : NEW_LINE INDENT if len ( string ) == 0 : NEW_LINE INDENT return NEW_LINE DEDENT temp = string [ 0 ] NEW_LINE reverse ( string [ 1 : ] ) NEW_LINE print ( temp , end = ' ' ) NEW_LINE DEDENT
string = " Geeks ▁ for ▁ Geeks " NEW_LINE reverse ( string ) NEW_LINE
box1 = 0 NEW_LINE
box2 = 0 NEW_LINE fact = [ 0 for i in range ( 11 ) ] NEW_LINE
def getProbability ( balls ) : NEW_LINE INDENT global box1 , box2 , fact NEW_LINE DEDENT
factorial ( 10 ) NEW_LINE
box2 = len ( balls ) NEW_LINE
K = 0 NEW_LINE
for i in range ( len ( balls ) ) : NEW_LINE INDENT K += balls [ i ] NEW_LINE DEDENT
if ( K % 2 == 1 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
all = comb ( K , K // 2 ) NEW_LINE
validPermutation = validPermutations ( K // 2 , balls , 0 , 0 ) NEW_LINE
return validPermutation / all NEW_LINE
def validPermutations ( n , balls , usedBalls , i ) : NEW_LINE INDENT global box1 , box2 , fact NEW_LINE DEDENT
if ( usedBalls == n ) : NEW_LINE
if ( box1 == box2 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT else : NEW_LINE INDENT return 0 NEW_LINE DEDENT
if ( i >= len ( balls ) ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
res = validPermutations ( n , balls , usedBalls , i + 1 ) NEW_LINE
box1 += 1 NEW_LINE
for j in range ( 1 , balls [ i ] + 1 ) : NEW_LINE
if ( j == balls [ i ] ) : NEW_LINE INDENT box2 -= 1 NEW_LINE DEDENT
combinations = comb ( balls [ i ] , j ) NEW_LINE
res += combinations * validPermutations ( n , balls , usedBalls + j , i + 1 ) NEW_LINE
box1 -= 1 NEW_LINE
box2 += 1 NEW_LINE return res NEW_LINE
def factorial ( N ) : NEW_LINE INDENT global box1 , box2 , fact NEW_LINE DEDENT
fact [ 0 ] = 1 NEW_LINE
for i in range ( 1 , N + 1 ) : NEW_LINE INDENT fact [ i ] = fact [ i - 1 ] * i NEW_LINE DEDENT
def comb ( n , r ) : NEW_LINE INDENT global box1 , box2 , fact NEW_LINE res = fact [ n ] // fact [ r ] NEW_LINE res //= fact [ n - r ] NEW_LINE return res NEW_LINE DEDENT
arr = [ 2 , 1 , 1 ] NEW_LINE N = 4 NEW_LINE
print ( getProbability ( arr ) ) NEW_LINE
from math import sin NEW_LINE
def polyarea ( n , r ) : NEW_LINE
if ( r < 0 and n < 0 ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
A = ( ( ( r * r * n ) * sin ( ( 360 / n ) * 3.14159 / 180 ) ) / 2 ) ; NEW_LINE return round ( A , 3 ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT r , n = 9 , 6 NEW_LINE print ( polyarea ( n , r ) ) NEW_LINE DEDENT
from collections import defaultdict NEW_LINE
def is_partition_possible ( n , x , y , w ) : NEW_LINE INDENT weight_at_x = defaultdict ( int ) NEW_LINE max_x = - 2e3 NEW_LINE min_x = 2e3 NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT new_x = x [ i ] - y [ i ] NEW_LINE max_x = max ( max_x , new_x ) NEW_LINE min_x = min ( min_x , new_x ) NEW_LINE DEDENT
weight_at_x [ new_x ] += w [ i ] NEW_LINE sum_till = [ ] NEW_LINE sum_till . append ( 0 ) NEW_LINE
for x in range ( min_x , max_x + 1 ) : NEW_LINE INDENT sum_till . append ( sum_till [ - 1 ] + weight_at_x [ x ] ) NEW_LINE DEDENT total_sum = sum_till [ - 1 ] NEW_LINE partition_possible = False NEW_LINE for i in range ( 1 , len ( sum_till ) ) : NEW_LINE INDENT if ( sum_till [ i ] == total_sum - sum_till [ i ] ) : NEW_LINE INDENT partition_possible = True NEW_LINE DEDENT DEDENT
if ( sum_till [ i - 1 ] == total_sum - sum_till [ i ] ) : NEW_LINE INDENT partition_possible = True NEW_LINE DEDENT if partition_possible : NEW_LINE print ( " YES " ) NEW_LINE else : NEW_LINE print ( " NO " ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 3 NEW_LINE x = [ - 1 , - 2 , 1 ] NEW_LINE y = [ 1 , 1 , - 1 ] NEW_LINE w = [ 3 , 1 , 4 ] NEW_LINE is_partition_possible ( n , x , y , w ) NEW_LINE DEDENT
def findPCSlope ( m ) : NEW_LINE INDENT return - 1.0 / m NEW_LINE DEDENT
m = 2.0 NEW_LINE print ( findPCSlope ( m ) ) NEW_LINE
import math NEW_LINE pi = 3.14159 NEW_LINE
def area_of_segment ( radius , angle ) : NEW_LINE
area_of_sector = pi * NEW_LINE INDENT ( radius * radius ) NEW_LINE * ( angle / 360 ) NEW_LINE DEDENT
area_of_triangle = 1 / 2 * NEW_LINE INDENT ( radius * radius ) * NEW_LINE math . sin ( ( angle * pi ) / 180 ) NEW_LINE DEDENT return area_of_sector - area_of_triangle ; NEW_LINE
radius = 10.0 NEW_LINE angle = 90.0 NEW_LINE print ( " Area ▁ of ▁ minor ▁ segment ▁ = " , area_of_segment ( radius , angle ) ) NEW_LINE print ( " Area ▁ of ▁ major ▁ segment ▁ = " , area_of_segment ( radius , ( 360 - angle ) ) ) NEW_LINE
def SectorArea ( radius , angle ) : NEW_LINE INDENT pi = 22 / 7 NEW_LINE if angle >= 360 : NEW_LINE INDENT print ( " Angle ▁ not ▁ possible " ) NEW_LINE return NEW_LINE DEDENT DEDENT
else : NEW_LINE INDENT sector = ( pi * radius ** 2 ) * ( angle / 360 ) NEW_LINE print ( sector ) NEW_LINE return NEW_LINE DEDENT
radius = 9 NEW_LINE angle = 60 NEW_LINE SectorArea ( radius , angle ) NEW_LINE
import math NEW_LINE
def PrimeFactor ( N ) : NEW_LINE INDENT ANS = dict ( ) NEW_LINE DEDENT
while N % 2 == 0 : NEW_LINE INDENT if 2 in ANS : NEW_LINE INDENT ANS [ 2 ] += 1 NEW_LINE DEDENT else : NEW_LINE INDENT ANS [ 2 ] = 1 NEW_LINE DEDENT DEDENT
N = N // 2 NEW_LINE
for i in range ( 3 , int ( math . sqrt ( N ) ) + 1 , 2 ) : NEW_LINE
while N % i == 0 : NEW_LINE INDENT if i in ANS : NEW_LINE INDENT ANS [ i ] += 1 NEW_LINE DEDENT else : NEW_LINE INDENT ANS [ i ] = 1 NEW_LINE DEDENT DEDENT
N = N // i NEW_LINE if 2 < N : NEW_LINE ANS [ N ] = 1 NEW_LINE return ANS NEW_LINE
def CountToMakeEqual ( X , Y ) : NEW_LINE
GCD = math . gcd ( X , Y ) NEW_LINE
newY = X // GCD NEW_LINE newX = Y // GCD NEW_LINE
primeX = PrimeFactor ( newX ) NEW_LINE primeY = PrimeFactor ( newY ) NEW_LINE
ans = 0 NEW_LINE
for factor in primeX : NEW_LINE INDENT if X % factor != 0 : NEW_LINE INDENT return - 1 NEW_LINE DEDENT ans += primeX [ factor ] NEW_LINE DEDENT
for factor in primeY : NEW_LINE INDENT if Y % factor != 0 : NEW_LINE INDENT return - 1 NEW_LINE DEDENT ans += primeY [ factor ] NEW_LINE DEDENT
return ans NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
X = 36 NEW_LINE Y = 48 NEW_LINE
ans = CountToMakeEqual ( X , Y ) NEW_LINE print ( ans ) NEW_LINE
from collections import deque NEW_LINE
def check ( Adj , Src , N , visited ) : NEW_LINE INDENT color = [ 0 ] * N NEW_LINE DEDENT
visited = [ True ] * Src NEW_LINE q = deque ( ) NEW_LINE
q . append ( Src ) NEW_LINE while ( len ( q ) > 0 ) : NEW_LINE
u = q . popleft ( ) NEW_LINE
Col = color [ u ] NEW_LINE
for x in Adj [ u ] : NEW_LINE
if ( visited [ x ] == True and color [ x ] == Col ) : NEW_LINE INDENT return False NEW_LINE DEDENT elif ( visited [ x ] == False ) : NEW_LINE
visited [ x ] = True NEW_LINE
q . append ( x ) NEW_LINE
color [ x ] = 1 - Col NEW_LINE
return True NEW_LINE
def addEdge ( Adj , u , v ) : NEW_LINE INDENT Adj [ u ] . append ( v ) NEW_LINE Adj [ v ] . append ( u ) NEW_LINE return Adj NEW_LINE DEDENT
def isPossible ( Arr , N ) : NEW_LINE
Adj = [ [ ] for i in range ( N ) ] NEW_LINE
for i in range ( N - 1 ) : NEW_LINE INDENT for j in range ( i + 1 , N ) : NEW_LINE DEDENT
if ( Arr [ i ] [ 0 ] < Arr [ j ] [ 1 ] or Arr [ i ] [ 1 ] > Arr [ j ] [ 0 ] ) : NEW_LINE INDENT continue NEW_LINE DEDENT
else : NEW_LINE INDENT if ( Arr [ i ] [ 2 ] == Arr [ j ] [ 2 ] ) : NEW_LINE DEDENT
Adj = addEdge ( Adj , i , j ) NEW_LINE
visited = [ False ] * N NEW_LINE
for i in range ( N ) : NEW_LINE INDENT if ( visited [ i ] == False and len ( Adj [ i ] ) > 0 ) : NEW_LINE DEDENT
if ( check ( Adj , i , N , visited ) == False ) : NEW_LINE INDENT print ( " No " ) NEW_LINE return NEW_LINE DEDENT
print ( " Yes " ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ [ 5 , 7 , 2 ] , [ 4 , 6 , 1 ] , [ 1 , 5 , 2 ] , [ 6 , 5 , 1 ] ] NEW_LINE N = len ( arr ) NEW_LINE isPossible ( arr , N ) NEW_LINE DEDENT
def lexNumbers ( n ) : NEW_LINE INDENT sol = [ ] NEW_LINE dfs ( 1 , n , sol ) NEW_LINE print ( " [ " , sol [ 0 ] , end = " " , sep = " " ) NEW_LINE for i in range ( 1 , n ) : NEW_LINE INDENT print ( " , ▁ " , sol [ i ] , end = " " , sep = " " ) print ( " ] " ) NEW_LINE DEDENT DEDENT def dfs ( temp , n , sol ) : NEW_LINE INDENT if ( temp > n ) : NEW_LINE INDENT return NEW_LINE DEDENT sol . append ( temp ) NEW_LINE dfs ( temp * 10 , n , sol ) NEW_LINE if ( temp % 10 != 9 ) : NEW_LINE INDENT dfs ( temp + 1 , n , sol ) NEW_LINE DEDENT DEDENT
n = 15 NEW_LINE lexNumbers ( n ) NEW_LINE
def minimumSwaps ( arr ) : NEW_LINE
count = 0 ; NEW_LINE i = 0 ; NEW_LINE while ( i < len ( arr ) ) : NEW_LINE
if ( arr [ i ] != i + 1 ) : NEW_LINE INDENT while ( arr [ i ] != i + 1 ) : NEW_LINE INDENT temp = 0 ; NEW_LINE DEDENT DEDENT
temp = arr [ arr [ i ] - 1 ] ; NEW_LINE arr [ arr [ i ] - 1 ] = arr [ i ] ; NEW_LINE arr [ i ] = temp ; NEW_LINE count += 1 ; NEW_LINE
i += 1 ; NEW_LINE return count ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 2 , 3 , 4 , 1 , 5 ] ; NEW_LINE DEDENT
print ( minimumSwaps ( arr ) ) ; NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , new_data ) : NEW_LINE INDENT self . data = new_data NEW_LINE self . next = None NEW_LINE self . prev = None NEW_LINE DEDENT DEDENT
def append ( head_ref , new_data ) : NEW_LINE
new_node = Node ( 0 ) NEW_LINE last = head_ref NEW_LINE
new_node . data = new_data NEW_LINE
new_node . next = None NEW_LINE
if ( head_ref == None ) : NEW_LINE INDENT new_node . prev = None NEW_LINE head_ref = new_node NEW_LINE return head_ref NEW_LINE DEDENT
while ( last . next != None ) : NEW_LINE INDENT last = last . next NEW_LINE DEDENT
last . next = new_node NEW_LINE
new_node . prev = last NEW_LINE return head_ref NEW_LINE
def printList ( node ) : NEW_LINE INDENT last = None NEW_LINE DEDENT
while ( node != None ) : NEW_LINE INDENT print ( node . data , end = " ▁ " ) NEW_LINE last = node NEW_LINE node = node . next NEW_LINE DEDENT
def mergeList ( p , q ) : NEW_LINE INDENT s = None NEW_LINE DEDENT
if ( p == None or q == None ) : NEW_LINE INDENT if ( p == None ) : NEW_LINE INDENT return q NEW_LINE DEDENT else : NEW_LINE INDENT return p NEW_LINE DEDENT DEDENT
if ( p . data < q . data ) : NEW_LINE INDENT p . prev = s NEW_LINE s = p NEW_LINE p = p . next NEW_LINE DEDENT else : NEW_LINE INDENT q . prev = s NEW_LINE s = q NEW_LINE q = q . next NEW_LINE DEDENT
head = s NEW_LINE while ( p != None and q != None ) : NEW_LINE INDENT if ( p . data < q . data ) : NEW_LINE DEDENT
s . next = p NEW_LINE p . prev = s NEW_LINE s = s . next NEW_LINE p = p . next NEW_LINE else : NEW_LINE
s . next = q NEW_LINE q . prev = s NEW_LINE s = s . next NEW_LINE q = q . next NEW_LINE
if ( p == None ) : NEW_LINE INDENT s . next = q NEW_LINE q . prev = s NEW_LINE DEDENT if ( q == None ) : NEW_LINE INDENT s . next = p NEW_LINE p . prev = s NEW_LINE DEDENT
return head NEW_LINE
def mergeAllList ( head , k ) : NEW_LINE INDENT finalList = None NEW_LINE i = 0 NEW_LINE while ( i < k ) : NEW_LINE DEDENT
finalList = mergeList ( finalList , head [ i ] ) NEW_LINE i = i + 1 NEW_LINE
return finalList NEW_LINE
k = 3 NEW_LINE head = [ 0 ] * k NEW_LINE i = 0 NEW_LINE
while ( i < k ) : NEW_LINE INDENT head [ i ] = None NEW_LINE i = i + 1 NEW_LINE DEDENT
head [ 0 ] = append ( head [ 0 ] , 1 ) NEW_LINE head [ 0 ] = append ( head [ 0 ] , 5 ) NEW_LINE head [ 0 ] = append ( head [ 0 ] , 9 ) NEW_LINE
head [ 1 ] = append ( head [ 1 ] , 2 ) NEW_LINE head [ 1 ] = append ( head [ 1 ] , 3 ) NEW_LINE head [ 1 ] = append ( head [ 1 ] , 7 ) NEW_LINE head [ 1 ] = append ( head [ 1 ] , 12 ) NEW_LINE
head [ 2 ] = append ( head [ 2 ] , 8 ) NEW_LINE head [ 2 ] = append ( head [ 2 ] , 11 ) NEW_LINE head [ 2 ] = append ( head [ 2 ] , 13 ) NEW_LINE head [ 2 ] = append ( head [ 2 ] , 18 ) NEW_LINE
finalList = mergeAllList ( head , k ) NEW_LINE
printList ( finalList ) NEW_LINE
def minIndex ( a , i , j ) : NEW_LINE INDENT if i == j : NEW_LINE INDENT return i NEW_LINE DEDENT DEDENT
k = minIndex ( a , i + 1 , j ) NEW_LINE
return ( i if a [ i ] < a [ k ] else k ) NEW_LINE
def recurSelectionSort ( a , n , index = 0 ) : NEW_LINE
if index == n : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
k = minIndex ( a , index , n - 1 ) NEW_LINE
if k != index : NEW_LINE INDENT a [ k ] , a [ index ] = a [ index ] , a [ k ] NEW_LINE DEDENT
a [ k ] , a [ index ] = a [ index ] , a [ k ] NEW_LINE
recurSelectionSort ( a , n , index + 1 ) NEW_LINE
arr = [ 3 , 1 , 5 , 2 , 7 , 0 ] NEW_LINE n = len ( arr ) NEW_LINE
recurSelectionSort ( arr , n ) NEW_LINE
for i in arr : NEW_LINE INDENT print ( i , end = ' ▁ ' ) NEW_LINE DEDENT
def insertionSortRecursive ( arr , n ) : NEW_LINE
if n <= 1 : NEW_LINE INDENT return NEW_LINE DEDENT
insertionSortRecursive ( arr , n - 1 ) NEW_LINE
last = arr [ n - 1 ] NEW_LINE j = n - 2 NEW_LINE
while ( j >= 0 and arr [ j ] > last ) : NEW_LINE INDENT arr [ j + 1 ] = arr [ j ] NEW_LINE j = j - 1 NEW_LINE DEDENT arr [ j + 1 ] = last NEW_LINE
arr = [ 12 , 11 , 13 , 5 , 6 ] NEW_LINE n = len ( arr ) NEW_LINE insertionSortRecursive ( arr , n ) NEW_LINE printArray ( arr , n ) NEW_LINE
class bubbleSort : NEW_LINE
def __init__ ( self , array ) : NEW_LINE INDENT self . array = array NEW_LINE self . length = len ( array ) NEW_LINE DEDENT def __str__ ( self ) : NEW_LINE INDENT return " ▁ " . join ( [ str ( x ) for x in self . array ] ) NEW_LINE DEDENT def bubbleSortRecursive ( self , n = None ) : NEW_LINE INDENT if n is None : NEW_LINE INDENT n = self . length NEW_LINE DEDENT DEDENT
if n == 1 : NEW_LINE INDENT return NEW_LINE DEDENT
for i in range ( n - 1 ) : NEW_LINE INDENT if self . array [ i ] > self . array [ i + 1 ] : NEW_LINE DEDENT
self . array [ i ] , self . array [ i + 1 ] = self . array [ i + 1 ] , self . array [ i ] NEW_LINE
self . bubbleSortRecursive ( n - 1 ) NEW_LINE
def maxSumAfterPartition ( arr , n ) : NEW_LINE
pos = [ ] NEW_LINE
neg = [ ] NEW_LINE
zero = 0 NEW_LINE
pos_sum = 0 NEW_LINE
neg_sum = 0 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT if ( arr [ i ] > 0 ) : NEW_LINE INDENT pos . append ( arr [ i ] ) NEW_LINE pos_sum += arr [ i ] NEW_LINE DEDENT elif ( arr [ i ] < 0 ) : NEW_LINE INDENT neg . append ( arr [ i ] ) NEW_LINE neg_sum += arr [ i ] NEW_LINE DEDENT else : NEW_LINE INDENT zero += 1 NEW_LINE DEDENT DEDENT
ans = 0 NEW_LINE
pos . sort ( ) NEW_LINE
neg . sort ( reverse = True ) NEW_LINE
if ( len ( pos ) > 0 and len ( neg ) > 0 ) : NEW_LINE INDENT ans = ( pos_sum - neg_sum ) NEW_LINE DEDENT elif ( len ( pos ) > 0 ) : NEW_LINE INDENT if ( zero > 0 ) : NEW_LINE DEDENT
ans = ( pos_sum ) NEW_LINE else : NEW_LINE
ans = ( pos_sum - 2 * pos [ 0 ] ) NEW_LINE else : NEW_LINE if ( zero > 0 ) : NEW_LINE
ans = ( - 1 * neg_sum ) NEW_LINE else : NEW_LINE
ans = ( neg [ 0 ] - ( neg_sum - neg [ 0 ] ) ) NEW_LINE return ans NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 1 , 2 , 3 , - 5 , - 7 ] NEW_LINE n = len ( arr ) NEW_LINE print ( maxSumAfterPartition ( arr , n ) ) NEW_LINE DEDENT
def MaxXOR ( arr , N ) : NEW_LINE
res = 0 NEW_LINE
for i in range ( N ) : NEW_LINE INDENT res |= arr [ i ] NEW_LINE DEDENT
return res NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 1 , 5 , 7 ] NEW_LINE N = len ( arr ) NEW_LINE print ( MaxXOR ( arr , N ) ) NEW_LINE DEDENT
def countEqual ( A , B , N ) : NEW_LINE
first = 0 NEW_LINE second = N - 1 NEW_LINE
count = 0 NEW_LINE while ( first < N and second >= 0 ) : NEW_LINE
if ( A [ first ] < B [ second ] ) : NEW_LINE
first += 1 NEW_LINE
elif ( B [ second ] < A [ first ] ) : NEW_LINE
second -= 1 NEW_LINE
else : NEW_LINE
count += 1 NEW_LINE
first += 1 NEW_LINE
second -= 1 NEW_LINE
return count NEW_LINE
A = [ 2 , 4 , 5 , 8 , 12 , 13 , 17 , 18 , 20 , 22 , 309 , 999 ] NEW_LINE B = [ 109 , 99 , 68 , 54 , 22 , 19 , 17 , 13 , 11 , 5 , 3 , 1 ] NEW_LINE N = len ( A ) NEW_LINE print ( countEqual ( A , B , N ) ) NEW_LINE
arr = [ 0 for i in range ( 100005 ) ] NEW_LINE
def isPalindrome ( N ) : NEW_LINE
temp = N NEW_LINE
res = 0 NEW_LINE
while ( temp != 0 ) : NEW_LINE INDENT rem = temp % 10 NEW_LINE res = res * 10 + rem NEW_LINE temp //= 10 NEW_LINE DEDENT
if ( res == N ) : NEW_LINE INDENT return True NEW_LINE DEDENT else : NEW_LINE INDENT return False NEW_LINE DEDENT
def sumOfDigits ( N ) : NEW_LINE
sum = 0 NEW_LINE while ( N != 0 ) : NEW_LINE
sum += N % 10 NEW_LINE
N //= 10 NEW_LINE
return sum NEW_LINE
def isPrime ( n ) : NEW_LINE
if ( n <= 1 ) : NEW_LINE INDENT return False NEW_LINE DEDENT
for i in range ( 2 , ( n // 2 ) + 1 , 1 ) : NEW_LINE
if ( n % i == 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT return True NEW_LINE
def precompute ( ) : NEW_LINE
for i in range ( 1 , 100001 , 1 ) : NEW_LINE
if ( isPalindrome ( i ) ) : NEW_LINE
sum = sumOfDigits ( i ) NEW_LINE
if ( isPrime ( sum ) ) : NEW_LINE INDENT arr [ i ] = 1 NEW_LINE DEDENT else : NEW_LINE INDENT arr [ i ] = 0 NEW_LINE DEDENT else : NEW_LINE arr [ i ] = 0 NEW_LINE
for i in range ( 1 , 100001 , 1 ) : NEW_LINE INDENT arr [ i ] = arr [ i ] + arr [ i - 1 ] NEW_LINE DEDENT
def countNumbers ( Q , N ) : NEW_LINE
precompute ( ) NEW_LINE
for i in range ( N ) : NEW_LINE
print ( arr [ Q [ i ] [ 1 ] ] - arr [ Q [ i ] [ 0 ] - 1 ] ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT Q = [ [ 5 , 9 ] , [ 1 , 101 ] ] NEW_LINE N = len ( Q ) NEW_LINE DEDENT
countNumbers ( Q , N ) NEW_LINE
def sum ( n ) : NEW_LINE INDENT sm = 0 NEW_LINE while ( n > 0 ) : NEW_LINE INDENT sm += n % 10 NEW_LINE n //= 10 NEW_LINE DEDENT return sm NEW_LINE DEDENT
def smallestNumber ( n , s ) : NEW_LINE
if ( sum ( n ) <= s ) : NEW_LINE INDENT return n NEW_LINE DEDENT
ans , k = n , 1 NEW_LINE for i in range ( 9 ) : NEW_LINE
digit = ( ans // k ) % 10 NEW_LINE
add = k * ( ( 10 - digit ) % 10 ) NEW_LINE ans += add NEW_LINE
if ( sum ( ans ) <= s ) : NEW_LINE INDENT break NEW_LINE DEDENT
k *= 10 NEW_LINE return ans NEW_LINE
n , s = 3 , 2 NEW_LINE
print ( smallestNumber ( n , s ) ) NEW_LINE
from collections import defaultdict NEW_LINE
def maxSubsequences ( arr , n ) -> int : NEW_LINE
m = defaultdict ( int ) NEW_LINE
maxCount = 0 NEW_LINE
count = 0 NEW_LINE for i in range ( 0 , n ) : NEW_LINE
if arr [ i ] in m . keys ( ) : NEW_LINE
count = m [ arr [ i ] ] NEW_LINE
if count > 1 : NEW_LINE
m [ arr [ i ] ] = count - 1 NEW_LINE
else : NEW_LINE INDENT m . pop ( arr [ i ] ) NEW_LINE DEDENT
if arr [ i ] - 1 > 0 : NEW_LINE INDENT m [ arr [ i ] - 1 ] += 1 NEW_LINE DEDENT else : NEW_LINE maxCount += 1 NEW_LINE
maxCount += 1 NEW_LINE
if arr [ i ] - 1 > 0 : NEW_LINE INDENT m [ arr [ i ] - 1 ] += 1 NEW_LINE DEDENT
return maxCount NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 5 NEW_LINE arr = [ 4 , 5 , 2 , 1 , 4 ] NEW_LINE print ( maxSubsequences ( arr , n ) ) NEW_LINE DEDENT
def removeOcc ( s , ch ) : NEW_LINE
for i in range ( len ( s ) ) : NEW_LINE
if ( s [ i ] == ch ) : NEW_LINE INDENT s = s [ 0 : i ] + s [ i + 1 : ] NEW_LINE break NEW_LINE DEDENT
for i in range ( len ( s ) - 1 , - 1 , - 1 ) : NEW_LINE
if ( s [ i ] == ch ) : NEW_LINE INDENT s = s [ 0 : i ] + s [ i + 1 : ] NEW_LINE break NEW_LINE DEDENT return s NEW_LINE
s = " hello ▁ world " NEW_LINE ch = ' l ' NEW_LINE print ( removeOcc ( s , ch ) ) NEW_LINE
import sys NEW_LINE
def minSteps ( N , increasing , decreasing ) : NEW_LINE
Min = sys . maxsize ; NEW_LINE
for i in increasing : NEW_LINE INDENT if ( Min > i ) : NEW_LINE INDENT Min = i ; NEW_LINE DEDENT DEDENT
Max = - sys . maxsize ; NEW_LINE
for i in decreasing : NEW_LINE INDENT if ( Max < i ) : NEW_LINE INDENT Max = i ; NEW_LINE DEDENT DEDENT
minSteps = max ( Max , N - Min ) ; NEW_LINE
print ( minSteps ) ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
N = 7 ; NEW_LINE
increasing = [ 3 , 5 ] ; NEW_LINE decreasing = [ 6 ] ; NEW_LINE
minSteps ( N , increasing , decreasing ) ; NEW_LINE
def solve ( P , n ) : NEW_LINE
arr = [ ] NEW_LINE arr . append ( 0 ) NEW_LINE for x in P : NEW_LINE INDENT arr . append ( x ) NEW_LINE DEDENT
cnt = 0 NEW_LINE for i in range ( 1 , n ) : NEW_LINE
if ( arr [ i ] == i ) : NEW_LINE INDENT arr [ i ] , arr [ i + 1 ] = arr [ i + 1 ] , arr [ i ] NEW_LINE cnt += 1 NEW_LINE DEDENT
if ( arr [ n ] == n ) : NEW_LINE
arr [ n - 1 ] , arr [ n ] = arr [ n ] , arr [ n - 1 ] NEW_LINE cnt += 1 NEW_LINE
print ( cnt ) NEW_LINE
N = 9 NEW_LINE
P = [ 1 , 2 , 4 , 9 , 5 , 8 , 7 , 3 , 6 ] NEW_LINE
solve ( P , N ) NEW_LINE
def SieveOfEratosthenes ( n , allPrimes ) : NEW_LINE
prime = [ True ] * ( n + 1 ) NEW_LINE p = 2 NEW_LINE while p * p <= n : NEW_LINE
if prime [ p ] == True : NEW_LINE
for i in range ( p * p , n + 1 , p ) : NEW_LINE INDENT prime [ i ] = False NEW_LINE DEDENT p += 1 NEW_LINE
for p in range ( 2 , n + 1 ) : NEW_LINE INDENT if prime [ p ] : NEW_LINE INDENT allPrimes . add ( p ) NEW_LINE DEDENT DEDENT
def countInterestingPrimes ( n ) : NEW_LINE
allPrimes = set ( ) NEW_LINE
SieveOfEratosthenes ( n , allPrimes ) NEW_LINE interestingPrimes = set ( ) NEW_LINE squares , quadruples = [ ] , [ ] NEW_LINE
i = 1 NEW_LINE while i * i <= n : NEW_LINE INDENT squares . append ( i * i ) NEW_LINE i += 1 NEW_LINE DEDENT
i = 1 NEW_LINE while i * i * i * i <= n : NEW_LINE INDENT quadruples . append ( i * i * i * i ) NEW_LINE i += 1 NEW_LINE DEDENT
for a in squares : NEW_LINE INDENT for b in quadruples : NEW_LINE INDENT if a + b in allPrimes : NEW_LINE INDENT interestingPrimes . add ( a + b ) NEW_LINE DEDENT DEDENT DEDENT
return len ( interestingPrimes ) NEW_LINE
N = 10 NEW_LINE print ( countInterestingPrimes ( N ) ) NEW_LINE
def isWaveArray ( arr , n ) : NEW_LINE INDENT result = True NEW_LINE DEDENT
if ( arr [ 1 ] > arr [ 0 ] and arr [ 1 ] > arr [ 2 ] ) : NEW_LINE INDENT for i in range ( 1 , n - 1 , 2 ) : NEW_LINE INDENT if ( arr [ i ] > arr [ i - 1 ] and arr [ i ] > arr [ i + 1 ] ) : NEW_LINE INDENT result = True NEW_LINE DEDENT else : NEW_LINE INDENT result = False NEW_LINE break NEW_LINE DEDENT DEDENT DEDENT
if ( result == True and n % 2 == 0 ) : NEW_LINE INDENT if ( arr [ n - 1 ] <= arr [ n - 2 ] ) : NEW_LINE INDENT result = False NEW_LINE DEDENT DEDENT elif ( arr [ 1 ] < arr [ 0 ] and arr [ 1 ] < arr [ 2 ] ) : NEW_LINE for i in range ( 1 , n - 1 , 2 ) : NEW_LINE INDENT if ( arr [ i ] < arr [ i - 1 ] and arr [ i ] < arr [ i + 1 ] ) : NEW_LINE INDENT result = True NEW_LINE DEDENT else : NEW_LINE INDENT result = False NEW_LINE break NEW_LINE DEDENT DEDENT
if ( result == True and n % 2 == 0 ) : NEW_LINE INDENT if ( arr [ n - 1 ] >= arr [ n - 2 ] ) : NEW_LINE INDENT result = False NEW_LINE DEDENT DEDENT return result NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
arr = [ 1 , 3 , 2 , 4 ] NEW_LINE n = len ( arr ) NEW_LINE if ( isWaveArray ( arr , n ) ) : NEW_LINE INDENT print ( " YES " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NO " ) NEW_LINE DEDENT
def countPossiblities ( arr , n ) : NEW_LINE
lastOccur = [ - 1 ] * 100000 NEW_LINE
dp = [ 0 ] * ( n + 1 ) NEW_LINE
dp [ 0 ] = 1 NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE INDENT curEle = arr [ i - 1 ] NEW_LINE DEDENT
dp [ i ] = dp [ i - 1 ] NEW_LINE
if ( lastOccur [ curEle ] != - 1 and lastOccur [ curEle ] < i - 1 ) : NEW_LINE INDENT dp [ i ] += dp [ lastOccur [ curEle ] ] NEW_LINE DEDENT
lastOccur [ curEle ] = i NEW_LINE
print ( dp [ n ] ) NEW_LINE
def maxSum ( arr , n , m ) : NEW_LINE
dp = [ [ 0 for i in range ( m + 1 ) ] for i in range ( 2 ) ] NEW_LINE
dp [ 0 ] [ m - 1 ] = arr [ 0 ] [ m - 1 ] NEW_LINE dp [ 1 ] [ m - 1 ] = arr [ 1 ] [ m - 1 ] NEW_LINE
for j in range ( m - 2 , - 1 , - 1 ) : NEW_LINE
for i in range ( 2 ) : NEW_LINE INDENT if ( i == 1 ) : NEW_LINE INDENT dp [ i ] [ j ] = max ( arr [ i ] [ j ] + dp [ 0 ] [ j + 1 ] , arr [ i ] [ j ] + dp [ 0 ] [ j + 2 ] ) NEW_LINE DEDENT else : NEW_LINE INDENT dp [ i ] [ j ] = max ( arr [ i ] [ j ] + dp [ 1 ] [ j + 1 ] , arr [ i ] [ j ] + dp [ 1 ] [ j + 2 ] ) NEW_LINE DEDENT DEDENT
print ( max ( dp [ 0 ] [ 0 ] , dp [ 1 ] [ 0 ] ) ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
arr = [ [ 1 , 50 , 21 , 5 ] , [ 2 , 10 , 10 , 5 ] ] NEW_LINE
N = len ( arr [ 0 ] ) NEW_LINE
maxSum ( arr , 2 , N ) NEW_LINE
def maxSum ( arr , n ) : NEW_LINE
r1 = r2 = 0 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT r1 , r2 = max ( r1 , r2 + arr [ 0 ] [ i ] ) , max ( r2 , r1 + arr [ 1 ] [ i ] ) NEW_LINE DEDENT
print ( max ( r1 , r2 ) ) NEW_LINE
arr = [ [ 1 , 50 , 21 , 5 ] , [ 2 , 10 , 10 , 5 ] ] NEW_LINE
n = len ( arr [ 0 ] ) NEW_LINE maxSum ( arr , n ) NEW_LINE
mod = 1e9 + 7 NEW_LINE mx = 1000000 NEW_LINE fact = [ 0 ] * ( mx + 1 ) NEW_LINE
def Calculate_factorial ( ) : NEW_LINE INDENT fact [ 0 ] = 1 NEW_LINE DEDENT
for i in range ( 1 , mx + 1 ) : NEW_LINE INDENT fact [ i ] = i * fact [ i - 1 ] NEW_LINE fact [ i ] %= mod NEW_LINE DEDENT
def UniModal_per ( a , b ) : NEW_LINE INDENT res = 1 NEW_LINE DEDENT
while ( b != 0 ) : NEW_LINE
if ( b % 2 != 0 ) : NEW_LINE INDENT res = res * a NEW_LINE DEDENT res %= mod NEW_LINE a = a * a NEW_LINE a %= mod NEW_LINE
b //= 2 NEW_LINE
return res NEW_LINE
def countPermutations ( n ) : NEW_LINE
Calculate_factorial ( ) NEW_LINE
uni_modal = UniModal_per ( 2 , n - 1 ) NEW_LINE
nonuni_modal = fact [ n ] - uni_modal NEW_LINE print ( int ( uni_modal ) , " " , int ( nonuni_modal ) ) NEW_LINE return NEW_LINE
N = 4 NEW_LINE
countPermutations ( N ) NEW_LINE
import sys NEW_LINE def longestSubseq ( s , length ) : NEW_LINE
ones = [ 0 for i in range ( length + 1 ) ] NEW_LINE zeroes = [ 0 for i in range ( length + 1 ) ] NEW_LINE
for i in range ( length ) : NEW_LINE
' NEW_LINE INDENT if ( s [ i ] == '1' ) : NEW_LINE INDENT ones [ i + 1 ] = ones [ i ] + 1 NEW_LINE zeroes [ i + 1 ] = zeroes [ i ] NEW_LINE DEDENT DEDENT
x += ones [ i ] NEW_LINE
x += ( zeroes [ j ] - zeroes [ i ] ) NEW_LINE
x += ( ones [ length ] - ones [ j ] ) NEW_LINE
answer = max ( answer , x ) NEW_LINE x = 0 NEW_LINE
print ( answer ) NEW_LINE
S = "10010010111100101" NEW_LINE length = len ( S ) NEW_LINE longestSubseq ( S , length ) NEW_LINE
MAX = 100 NEW_LINE
def largestSquare ( matrix , R , C , q_i , q_j , K , Q ) : NEW_LINE
for q in range ( Q ) : NEW_LINE INDENT i = q_i [ q ] NEW_LINE j = q_j [ q ] NEW_LINE min_dist = min ( min ( i , j ) , min ( R - i - 1 , C - j - 1 ) ) NEW_LINE ans = - 1 NEW_LINE for k in range ( min_dist + 1 ) : NEW_LINE INDENT count = 0 NEW_LINE DEDENT DEDENT
for row in range ( i - k , i + k + 1 ) : NEW_LINE INDENT for col in range ( j - k , j + k + 1 ) : NEW_LINE INDENT count += matrix [ row ] [ col ] NEW_LINE DEDENT DEDENT
if count > K : NEW_LINE INDENT break NEW_LINE DEDENT ans = 2 * k + 1 NEW_LINE print ( ans ) NEW_LINE
matrix = [ [ 1 , 0 , 1 , 0 , 0 ] , [ 1 , 0 , 1 , 1 , 1 ] , [ 1 , 1 , 1 , 1 , 1 ] , [ 1 , 0 , 0 , 1 , 0 ] ] NEW_LINE K = 9 NEW_LINE Q = 1 NEW_LINE q_i = [ 1 ] NEW_LINE q_j = [ 2 ] NEW_LINE largestSquare ( matrix , 4 , 5 , q_i , q_j , K , Q ) NEW_LINE
def largestSquare ( matrix , R , C , q_i , q_j , K , Q ) : NEW_LINE INDENT countDP = [ [ 0 for x in range ( C ) ] for x in range ( R ) ] NEW_LINE DEDENT
countDP [ 0 ] [ 0 ] = matrix [ 0 ] [ 0 ] NEW_LINE for i in range ( 1 , R ) : NEW_LINE INDENT countDP [ i ] [ 0 ] = ( countDP [ i - 1 ] [ 0 ] + matrix [ i ] [ 0 ] ) NEW_LINE DEDENT for j in range ( 1 , C ) : NEW_LINE INDENT countDP [ 0 ] [ j ] = ( countDP [ 0 ] [ j - 1 ] + matrix [ 0 ] [ j ] ) NEW_LINE DEDENT for i in range ( 1 , R ) : NEW_LINE INDENT for j in range ( 1 , C ) : NEW_LINE INDENT countDP [ i ] [ j ] = ( matrix [ i ] [ j ] + countDP [ i - 1 ] [ j ] + countDP [ i ] [ j - 1 ] - countDP [ i - 1 ] [ j - 1 ] ) NEW_LINE DEDENT DEDENT
for q in range ( 0 , Q ) : NEW_LINE INDENT i = q_i [ q ] NEW_LINE j = q_j [ q ] NEW_LINE DEDENT
min_dist = min ( i , j , R - i - 1 , C - j - 1 ) NEW_LINE ans = - 1 NEW_LINE for k in range ( 0 , min_dist + 1 ) : NEW_LINE INDENT x1 = i - k NEW_LINE x2 = i + k NEW_LINE y1 = j - k NEW_LINE y2 = j + k NEW_LINE DEDENT
count = countDP [ x2 ] [ y2 ] ; NEW_LINE if ( x1 > 0 ) : NEW_LINE INDENT count -= countDP [ x1 - 1 ] [ y2 ] NEW_LINE DEDENT if ( y1 > 0 ) : NEW_LINE INDENT count -= countDP [ x2 ] [ y1 - 1 ] NEW_LINE DEDENT if ( x1 > 0 and y1 > 0 ) : NEW_LINE INDENT count += countDP [ x1 - 1 ] [ y1 - 1 ] NEW_LINE DEDENT if ( count > K ) : NEW_LINE INDENT break NEW_LINE DEDENT ans = 2 * k + 1 NEW_LINE print ( ans ) NEW_LINE
matrix = [ [ 1 , 0 , 1 , 0 , 0 ] , [ 1 , 0 , 1 , 1 , 1 ] , [ 1 , 1 , 1 , 1 , 1 ] , [ 1 , 0 , 0 , 1 , 0 ] ] NEW_LINE K = 9 NEW_LINE Q = 1 NEW_LINE q_i = [ 1 ] NEW_LINE q_j = [ 2 ] NEW_LINE largestSquare ( matrix , 4 , 5 , q_i , q_j , K , Q ) NEW_LINE
def MinCost ( arr , n ) : NEW_LINE
dp = [ [ 0 for i in range ( n + 5 ) ] for i in range ( n + 5 ) ] NEW_LINE sum = [ [ 0 for i in range ( n + 5 ) ] for i in range ( n + 5 ) ] NEW_LINE
for i in range ( n ) : NEW_LINE INDENT k = arr [ i ] NEW_LINE for j in range ( i , n ) : NEW_LINE INDENT if ( i == j ) : NEW_LINE INDENT sum [ i ] [ j ] = k NEW_LINE DEDENT else : NEW_LINE INDENT k += arr [ j ] NEW_LINE sum [ i ] [ j ] = k NEW_LINE DEDENT DEDENT DEDENT
for i in range ( n - 1 , - 1 , - 1 ) : NEW_LINE
for j in range ( i , n ) : NEW_LINE INDENT dp [ i ] [ j ] = 10 ** 9 NEW_LINE DEDENT
if ( i == j ) : NEW_LINE INDENT dp [ i ] [ j ] = 0 NEW_LINE DEDENT else : NEW_LINE INDENT for k in range ( i , j ) : NEW_LINE INDENT dp [ i ] [ j ] = min ( dp [ i ] [ j ] , dp [ i ] [ k ] + dp [ k + 1 ] [ j ] + sum [ i ] [ j ] ) NEW_LINE DEDENT DEDENT return dp [ 0 ] [ n - 1 ] NEW_LINE
arr = [ 7 , 6 , 8 , 6 , 1 , 1 ] NEW_LINE n = len ( arr ) NEW_LINE print ( MinCost ( arr , n ) ) NEW_LINE
def f ( i , state , A , dp , N ) : NEW_LINE INDENT if i >= N : NEW_LINE INDENT return 0 NEW_LINE DEDENT DEDENT
elif dp [ i ] [ state ] != - 1 : NEW_LINE INDENT return dp [ i ] [ state ] NEW_LINE DEDENT
else : NEW_LINE INDENT if i == N - 1 : NEW_LINE INDENT dp [ i ] [ state ] = 1 NEW_LINE DEDENT elif state == 1 and A [ i ] > A [ i + 1 ] : NEW_LINE INDENT dp [ i ] [ state ] = 1 NEW_LINE DEDENT elif state == 2 and A [ i ] < A [ i + 1 ] : NEW_LINE INDENT dp [ i ] [ state ] = 1 NEW_LINE DEDENT elif state == 1 and A [ i ] <= A [ i + 1 ] : NEW_LINE INDENT dp [ i ] [ state ] = 1 + f ( i + 1 , 2 , A , dp , N ) NEW_LINE DEDENT elif state == 2 and A [ i ] >= A [ i + 1 ] : NEW_LINE INDENT dp [ i ] [ state ] = 1 + f ( i + 1 , 1 , A , dp , N ) NEW_LINE DEDENT return dp [ i ] [ state ] NEW_LINE DEDENT
def maxLenSeq ( A , N ) : NEW_LINE
dp = [ [ - 1 , - 1 , - 1 ] for i in range ( 1000 ) ] NEW_LINE
for i in range ( N ) : NEW_LINE INDENT tmp = f ( i , 1 , A , dp , N ) NEW_LINE tmp = f ( i , 2 , A , dp , N ) NEW_LINE DEDENT
ans = - 1 NEW_LINE for i in range ( N ) : NEW_LINE
y = dp [ i ] [ 1 ] NEW_LINE if ( i + y ) >= N : NEW_LINE INDENT ans = max ( ans , dp [ i ] [ 1 ] + 1 ) NEW_LINE DEDENT
elif y % 2 == 0 : NEW_LINE INDENT ans = max ( ans , dp [ i ] [ 1 ] + 1 + dp [ i + y ] [ 2 ] ) NEW_LINE DEDENT
elif y % 2 == 1 : NEW_LINE INDENT ans = max ( ans , dp [ i ] [ 1 ] + 1 + dp [ i + y ] [ 1 ] ) NEW_LINE DEDENT return ans NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT A = [ 1 , 10 , 3 , 20 , 25 , 24 ] NEW_LINE n = len ( A ) NEW_LINE print ( maxLenSeq ( A , n ) ) NEW_LINE DEDENT
import math as mt NEW_LINE
def MaxGCD ( a , n ) : NEW_LINE
Prefix = [ 0 for i in range ( n + 2 ) ] NEW_LINE Suffix = [ 0 for i in range ( n + 2 ) ] NEW_LINE
Prefix [ 1 ] = a [ 0 ] NEW_LINE for i in range ( 2 , n + 1 ) : NEW_LINE INDENT Prefix [ i ] = mt . gcd ( Prefix [ i - 1 ] , a [ i - 1 ] ) NEW_LINE DEDENT
Suffix [ n ] = a [ n - 1 ] NEW_LINE
for i in range ( n - 1 , 0 , - 1 ) : NEW_LINE INDENT Suffix [ i ] = mt . gcd ( Suffix [ i + 1 ] , a [ i - 1 ] ) NEW_LINE DEDENT
ans = max ( Suffix [ 2 ] , Prefix [ n - 1 ] ) NEW_LINE
for i in range ( 2 , n ) : NEW_LINE INDENT ans = max ( ans , mt . gcd ( Prefix [ i - 1 ] , Suffix [ i + 1 ] ) ) NEW_LINE DEDENT
return ans NEW_LINE
a = [ 14 , 17 , 28 , 70 ] NEW_LINE n = len ( a ) NEW_LINE print ( MaxGCD ( a , n ) ) NEW_LINE
import numpy as np NEW_LINE right = 3 ; NEW_LINE left = 6 ; NEW_LINE dp = np . ones ( ( left , right ) ) NEW_LINE dp = - 1 * dp NEW_LINE
def findSubarraySum ( ind , flips , n , a , k ) : NEW_LINE
if ( flips > k ) : NEW_LINE INDENT return - 1e9 ; NEW_LINE DEDENT
if ( ind == n ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT
if ( dp [ ind ] [ flips ] != - 1 ) : NEW_LINE INDENT return dp [ ind ] [ flips ] ; NEW_LINE DEDENT
ans = 0 ; NEW_LINE
ans = max ( 0 , a [ ind ] + findSubarraySum ( ind + 1 , flips , n , a , k ) ) ; NEW_LINE ans = max ( ans , - a [ ind ] + findSubarraySum ( ind + 1 , flips + 1 , n , a , k ) ) ; NEW_LINE
dp [ ind ] [ flips ] = ans ; NEW_LINE return dp [ ind ] [ flips ] ; NEW_LINE
def findMaxSubarraySum ( a , n , k ) : NEW_LINE
ans = - 1e9 ; NEW_LINE
for i in range ( n ) : NEW_LINE INDENT ans = max ( ans , findSubarraySum ( i , 0 , n , a , k ) ) ; NEW_LINE DEDENT
if ans == 0 and k == 0 : NEW_LINE return max ( a ) ; NEW_LINE return ans ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = [ - 1 , - 2 , - 100 , - 10 ] ; NEW_LINE n = len ( a ) ; NEW_LINE k = 1 ; NEW_LINE print ( findMaxSubarraySum ( a , n , k ) ) ; NEW_LINE DEDENT
mod = 1000000007 ; NEW_LINE
def sumOddFibonacci ( n ) : NEW_LINE INDENT Sum = [ 0 ] * ( n + 1 ) ; NEW_LINE DEDENT
Sum [ 0 ] = 0 ; NEW_LINE Sum [ 1 ] = 1 ; NEW_LINE Sum [ 2 ] = 2 ; NEW_LINE Sum [ 3 ] = 5 ; NEW_LINE Sum [ 4 ] = 10 ; NEW_LINE Sum [ 5 ] = 23 ; NEW_LINE for i in range ( 6 , n + 1 ) : NEW_LINE INDENT Sum [ i ] = ( ( Sum [ i - 1 ] + ( 4 * Sum [ i - 2 ] ) % mod - ( 4 * Sum [ i - 3 ] ) % mod + mod ) % mod + ( Sum [ i - 4 ] - Sum [ i - 5 ] + mod ) % mod ) % mod ; NEW_LINE DEDENT return Sum [ n ] ; NEW_LINE
n = 6 ; NEW_LINE print ( sumOddFibonacci ( n ) ) ; NEW_LINE
def fun ( marks , n ) : NEW_LINE
dp = [ 1 for i in range ( 0 , n ) ] NEW_LINE for i in range ( 0 , n - 1 ) : NEW_LINE
if marks [ i ] > marks [ i + 1 ] : NEW_LINE INDENT temp = i NEW_LINE while True : NEW_LINE INDENT if marks [ temp ] > marks [ temp + 1 ] and temp >= 0 : NEW_LINE INDENT if dp [ temp ] > dp [ temp + 1 ] : NEW_LINE INDENT temp -= 1 NEW_LINE continue NEW_LINE DEDENT else : NEW_LINE INDENT dp [ temp ] = dp [ temp + 1 ] + 1 NEW_LINE temp -= 1 NEW_LINE DEDENT DEDENT else : NEW_LINE INDENT break NEW_LINE DEDENT DEDENT DEDENT
elif marks [ i ] < marks [ i + 1 ] : NEW_LINE INDENT dp [ i + 1 ] = dp [ i ] + 1 NEW_LINE DEDENT return ( sum ( dp ) ) NEW_LINE
n = 6 NEW_LINE
marks = [ 1 , 4 , 5 , 2 , 2 , 1 ] NEW_LINE
print ( fun ( marks , n ) ) NEW_LINE
def solve ( N , K ) : NEW_LINE
combo = [ 0 ] * ( N + 1 ) NEW_LINE
combo [ 0 ] = 1 NEW_LINE
for i in range ( 1 , K + 1 ) : NEW_LINE
for j in range ( 0 , N + 1 ) : NEW_LINE
if j >= i : NEW_LINE
combo [ j ] += combo [ j - i ] NEW_LINE
return combo [ N ] NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
N , K = 29 , 5 NEW_LINE print ( solve ( N , K ) ) NEW_LINE
def computeLIS ( circBuff , start , end , n ) : NEW_LINE INDENT LIS = [ 0 for i in range ( end ) ] NEW_LINE DEDENT
for i in range ( start , end ) : NEW_LINE INDENT LIS [ i ] = 1 NEW_LINE DEDENT
for i in range ( start + 1 , end ) : NEW_LINE
for j in range ( start , i ) : NEW_LINE INDENT if ( circBuff [ i ] > circBuff [ j ] and LIS [ i ] < LIS [ j ] + 1 ) : NEW_LINE INDENT LIS [ i ] = LIS [ j ] + 1 NEW_LINE DEDENT DEDENT
res = - 100000 NEW_LINE for i in range ( start , end ) : NEW_LINE INDENT res = max ( res , LIS [ i ] ) NEW_LINE DEDENT return res NEW_LINE
def LICS ( arr , n ) : NEW_LINE
circBuff = [ 0 for i in range ( 2 * n ) ] NEW_LINE for i in range ( n ) : NEW_LINE INDENT circBuff [ i ] = arr [ i ] NEW_LINE DEDENT for i in range ( n , 2 * n ) : NEW_LINE INDENT circBuff [ i ] = arr [ i - n ] NEW_LINE DEDENT
res = - 100000 NEW_LINE for i in range ( n ) : NEW_LINE INDENT res = max ( computeLIS ( circBuff , i , i + n , n ) , res ) NEW_LINE DEDENT return res NEW_LINE
arr = [ 1 , 4 , 6 , 2 , 3 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Length ▁ of ▁ LICS ▁ is " , LICS ( arr , n ) ) NEW_LINE
def binomialCoeff ( n , k ) : NEW_LINE INDENT C = [ 0 ] * ( k + 1 ) NEW_LINE C [ 0 ] = 1 NEW_LINE DEDENT
for i in range ( 1 , n + 1 ) : NEW_LINE INDENT j = min ( i , k ) NEW_LINE while ( j > 0 ) : NEW_LINE INDENT C [ j ] = C [ j ] + C [ j - 1 ] NEW_LINE j -= 1 NEW_LINE DEDENT DEDENT return C [ k ] NEW_LINE
n = 3 NEW_LINE m = 2 NEW_LINE print ( " Number ▁ of ▁ Paths : " , binomialCoeff ( n + m , n ) ) NEW_LINE
def LCIS ( arr1 , n , arr2 , m ) : NEW_LINE
table = [ 0 ] * m NEW_LINE for j in range ( m ) : NEW_LINE INDENT table [ j ] = 0 NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE
current = 0 NEW_LINE
for j in range ( m ) : NEW_LINE
if ( arr1 [ i ] == arr2 [ j ] ) : NEW_LINE INDENT if ( current + 1 > table [ j ] ) : NEW_LINE INDENT table [ j ] = current + 1 NEW_LINE DEDENT DEDENT
if ( arr1 [ i ] > arr2 [ j ] ) : NEW_LINE INDENT if ( table [ j ] > current ) : NEW_LINE INDENT current = table [ j ] NEW_LINE DEDENT DEDENT
result = 0 NEW_LINE for i in range ( m ) : NEW_LINE INDENT if ( table [ i ] > result ) : NEW_LINE INDENT result = table [ i ] NEW_LINE DEDENT DEDENT return result NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr1 = [ 3 , 4 , 9 , 1 ] NEW_LINE arr2 = [ 5 , 3 , 8 , 9 , 10 , 2 , 1 ] NEW_LINE n = len ( arr1 ) NEW_LINE m = len ( arr2 ) NEW_LINE print ( " Length ▁ of ▁ LCIS ▁ is " , LCIS ( arr1 , n , arr2 , m ) ) NEW_LINE DEDENT
import sys NEW_LINE
def longComPre ( arr , N ) : NEW_LINE
freq = [ [ 0 for i in range ( 256 ) ] for i in range ( N ) ] NEW_LINE
for i in range ( N ) : NEW_LINE
M = len ( arr [ i ] ) NEW_LINE
for j in range ( M ) : NEW_LINE
freq [ i ] [ ord ( arr [ i ] [ j ] ) ] += 1 NEW_LINE
maxLen = 0 NEW_LINE
for j in range ( 256 ) : NEW_LINE
minRowVal = sys . maxsize NEW_LINE
for i in range ( N ) : NEW_LINE
minRowVal = min ( minRowVal , freq [ i ] [ j ] ) NEW_LINE
maxLen += minRowVal NEW_LINE return maxLen NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ " aabdc " , " abcd " , " aacd " ] NEW_LINE N = 3 NEW_LINE print ( longComPre ( arr , N ) ) NEW_LINE DEDENT
MAX_CHAR = 26 NEW_LINE
def removeChars ( arr , k ) : NEW_LINE
hash = [ 0 ] * MAX_CHAR NEW_LINE
n = len ( arr ) NEW_LINE for i in range ( n ) : NEW_LINE INDENT hash [ ord ( arr [ i ] ) - ord ( ' a ' ) ] += 1 NEW_LINE DEDENT
ans = " " NEW_LINE
index = 0 NEW_LINE for i in range ( n ) : NEW_LINE
if ( hash [ ord ( arr [ i ] ) - ord ( ' a ' ) ] != k ) : NEW_LINE INDENT ans += arr [ i ] NEW_LINE DEDENT return ans NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT str = " geeksforgeeks " NEW_LINE k = 2 NEW_LINE DEDENT
print ( removeChars ( str , k ) ) NEW_LINE
def sub_segments ( string , n ) : NEW_LINE INDENT l = len ( string ) NEW_LINE for x in range ( 0 , l , n ) : NEW_LINE INDENT newlist = string [ x : x + n ] NEW_LINE DEDENT DEDENT
arr = [ ] NEW_LINE for y in newlist : NEW_LINE
if y not in arr : NEW_LINE INDENT arr . append ( y ) NEW_LINE DEDENT print ( ' ' . join ( arr ) ) NEW_LINE
string = " geeksforgeeksgfg " NEW_LINE n = 4 NEW_LINE sub_segments ( string , n ) NEW_LINE
def findWord ( c , n ) : NEW_LINE INDENT co = 0 NEW_LINE DEDENT
s = [ 0 ] * n NEW_LINE for i in range ( n ) : NEW_LINE INDENT if ( i < n / 2 ) : NEW_LINE INDENT co += 1 NEW_LINE DEDENT else : NEW_LINE INDENT co = n - i NEW_LINE DEDENT DEDENT
if ( ord ( c [ i ] ) + co <= 122 ) : NEW_LINE INDENT s [ i ] = chr ( ord ( c [ i ] ) + co ) NEW_LINE DEDENT else : NEW_LINE INDENT s [ i ] = chr ( ord ( c [ i ] ) + co - 26 ) NEW_LINE DEDENT print ( * s , sep = " " ) NEW_LINE
s = " abcd " NEW_LINE findWord ( s , len ( s ) ) NEW_LINE
def equalIgnoreCase ( str1 , str2 ) : NEW_LINE INDENT i = 0 NEW_LINE DEDENT
len1 = len ( str1 ) NEW_LINE
len2 = len ( str2 ) NEW_LINE
if ( len1 != len2 ) : NEW_LINE INDENT return False NEW_LINE DEDENT
while ( i < len1 ) : NEW_LINE
if ( str1 [ i ] == str2 [ i ] ) : NEW_LINE INDENT i += 1 NEW_LINE DEDENT
elif ( ( ( str1 [ i ] >= ' a ' and str1 [ i ] <= ' z ' ) or ( str1 [ i ] >= ' A ' and str1 [ i ] <= ' Z ' ) ) == False ) : NEW_LINE INDENT return False NEW_LINE DEDENT
elif ( ( ( str2 [ i ] >= ' a ' and str2 [ i ] <= ' z ' ) or ( str2 [ i ] >= ' A ' and str2 [ i ] <= ' Z ' ) ) == False ) : NEW_LINE INDENT return False NEW_LINE DEDENT
else : NEW_LINE
if ( str1 [ i ] >= ' a ' and str1 [ i ] <= ' z ' ) : NEW_LINE INDENT if ( ord ( str1 [ i ] ) - 32 != ord ( str2 [ i ] ) ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT elif ( str1 [ i ] >= ' A ' and str1 [ i ] <= ' Z ' ) : NEW_LINE INDENT if ( ord ( str1 [ i ] ) + 32 != ord ( str2 [ i ] ) ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT
i += 1 NEW_LINE
return True NEW_LINE
def equalIgnoreCaseUtil ( str1 , str2 ) : NEW_LINE INDENT res = equalIgnoreCase ( str1 , str2 ) NEW_LINE if ( res == True ) : NEW_LINE INDENT print ( " Same " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Not ▁ Same " ) NEW_LINE DEDENT DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT str1 = " Geeks " NEW_LINE str2 = " geeks " NEW_LINE equalIgnoreCaseUtil ( str1 , str2 ) NEW_LINE str1 = " Geek " NEW_LINE str2 = " geeksforgeeks " NEW_LINE equalIgnoreCaseUtil ( str1 , str2 ) NEW_LINE DEDENT
def maxValue ( a , b ) : NEW_LINE
b = sorted ( b ) NEW_LINE bi = [ i for i in b ] NEW_LINE ai = [ i for i in a ] NEW_LINE n = len ( a ) NEW_LINE m = len ( b ) NEW_LINE
j = m - 1 NEW_LINE for i in range ( n ) : NEW_LINE
if ( j < 0 ) : NEW_LINE INDENT break NEW_LINE DEDENT if ( bi [ j ] > ai [ i ] ) : NEW_LINE INDENT ai [ i ] = bi [ j ] NEW_LINE DEDENT
j -= 1 NEW_LINE
x = " " . join ( ai ) NEW_LINE return x NEW_LINE
a = "1234" NEW_LINE b = "4321" NEW_LINE print ( maxValue ( a , b ) ) NEW_LINE
def checkIfUnequal ( n , q ) : NEW_LINE
s1 = str ( n ) NEW_LINE a = [ 0 for i in range ( 26 ) ] NEW_LINE
for i in range ( 0 , len ( s1 ) , 1 ) : NEW_LINE INDENT a [ ord ( s1 [ i ] ) - ord ( '0' ) ] += 1 NEW_LINE DEDENT
prod = n * q NEW_LINE
s2 = str ( prod ) NEW_LINE
for i in range ( 0 , len ( s2 ) , 1 ) : NEW_LINE
if ( a [ ord ( s2 [ i ] ) - ord ( '0' ) ] ) : NEW_LINE INDENT return False NEW_LINE DEDENT
return True NEW_LINE
def countInRange ( l , r , q ) : NEW_LINE INDENT count = 0 NEW_LINE for i in range ( l , r + 1 , 1 ) : NEW_LINE DEDENT
if ( checkIfUnequal ( i , q ) ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT return count NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT l = 10 NEW_LINE r = 12 NEW_LINE q = 2 NEW_LINE DEDENT
print ( countInRange ( l , r , q ) ) NEW_LINE
def is_possible ( s ) : NEW_LINE
l = len ( s ) NEW_LINE one = 0 NEW_LINE zero = 0 NEW_LINE for i in range ( 0 , l ) : NEW_LINE
if ( s [ i ] == '0' ) : NEW_LINE INDENT zero += 1 NEW_LINE DEDENT
else : NEW_LINE INDENT one += 1 NEW_LINE DEDENT
if ( l % 2 == 0 ) : NEW_LINE INDENT return ( one == zero ) NEW_LINE DEDENT
else : NEW_LINE INDENT return ( abs ( one - zero ) == 1 ) NEW_LINE DEDENT
limit = 255 NEW_LINE def countFreq ( Str ) : NEW_LINE
count = [ 0 ] * ( limit + 1 ) NEW_LINE
for i in range ( len ( Str ) ) : NEW_LINE INDENT count [ ord ( Str [ i ] ) ] += 1 NEW_LINE DEDENT for i in range ( limit + 1 ) : NEW_LINE if ( count [ i ] > 0 ) : NEW_LINE INDENT print ( chr ( i ) , count [ i ] ) NEW_LINE DEDENT
/ * Driver Code * / NEW_LINE Str = " GeeksforGeeks " NEW_LINE countFreq ( Str ) NEW_LINE
def countEvenOdd ( arr , n , K ) : NEW_LINE INDENT even = 0 ; odd = 0 ; NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE
x = bin ( arr [ i ] ) . count ( '1' ) ; NEW_LINE if ( x % 2 == 0 ) : NEW_LINE INDENT even += 1 ; NEW_LINE DEDENT else : NEW_LINE INDENT odd += 1 ; NEW_LINE DEDENT
y = bin ( K ) . count ( '1' ) ; NEW_LINE
if ( y & 1 ) : NEW_LINE INDENT print ( " Even ▁ = " , odd , " , ▁ Odd ▁ = " , even ) ; NEW_LINE DEDENT
else : NEW_LINE INDENT print ( " Even ▁ = " , even , " , ▁ Odd ▁ = " , odd ) ; NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 4 , 2 , 15 , 9 , 8 , 8 ] ; NEW_LINE K = 3 ; NEW_LINE n = len ( arr ) ; NEW_LINE DEDENT
countEvenOdd ( arr , n , K ) ; NEW_LINE
import math NEW_LINE
def convert ( s ) : NEW_LINE INDENT n = len ( s ) NEW_LINE s1 = " " NEW_LINE s1 = s1 + s [ 0 ] . lower ( ) NEW_LINE i = 1 NEW_LINE while i < n : NEW_LINE DEDENT
if ( s [ i ] == ' ▁ ' and i <= n ) : NEW_LINE
s1 = s1 + " ▁ " + ( s [ i + 1 ] ) . lower ( ) NEW_LINE i = i + 1 NEW_LINE
else : NEW_LINE INDENT s1 = s1 + ( s [ i ] ) . upper ( ) NEW_LINE DEDENT i = i + 1 NEW_LINE
return s1 NEW_LINE
str = " I ▁ get ▁ intern ▁ at ▁ geeksforgeeks " NEW_LINE print ( convert ( str ) ) NEW_LINE
import math NEW_LINE
def reverse ( num ) : NEW_LINE INDENT rev_num = 0 NEW_LINE while ( num > 0 ) : NEW_LINE INDENT rev_num = rev_num * 10 + num % 10 NEW_LINE num = num // 10 NEW_LINE DEDENT return rev_num NEW_LINE DEDENT
def properDivSum ( num ) : NEW_LINE
result = 0 NEW_LINE
' NEW_LINE INDENT for i in range ( 2 , ( int ) ( math . sqrt ( num ) ) + 1 ) : NEW_LINE DEDENT
' NEW_LINE INDENT if ( num % i == 0 ) : NEW_LINE DEDENT
if ( i == ( num // i ) ) : NEW_LINE INDENT result += i NEW_LINE DEDENT else : NEW_LINE INDENT result += ( i + num / i ) NEW_LINE DEDENT
return ( result + 1 ) NEW_LINE def isTcefrep ( n ) : NEW_LINE return properDivSum ( n ) == reverse ( n ) ; NEW_LINE
N = 6 NEW_LINE
if ( isTcefrep ( N ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
def reverse ( s ) : NEW_LINE INDENT if len ( s ) == 0 : NEW_LINE INDENT return s NEW_LINE DEDENT else : NEW_LINE INDENT return reverse ( s [ 1 : ] ) + s [ 0 ] NEW_LINE DEDENT DEDENT def findNthNo ( n ) : NEW_LINE INDENT res = " " ; NEW_LINE while ( n >= 1 ) : NEW_LINE DEDENT
if ( n & 1 ) : NEW_LINE INDENT res = res + "3" ; NEW_LINE n = ( int ) ( ( n - 1 ) / 2 ) ; NEW_LINE DEDENT
else : NEW_LINE INDENT res = res + "5" ; NEW_LINE n = ( int ) ( ( n - 2 ) / 2 ) ; NEW_LINE DEDENT
return reverse ( res ) ; NEW_LINE
n = 5 ; NEW_LINE print ( findNthNo ( n ) ) ; NEW_LINE
import math NEW_LINE
def findNthNonSquare ( n ) : NEW_LINE
x = n ; NEW_LINE
ans = x + math . floor ( 0.5 + math . sqrt ( x ) ) ; NEW_LINE return int ( ans ) ; NEW_LINE
n = 16 ; NEW_LINE
print ( " The " , n , " th ▁ Non - Square ▁ number ▁ is " , findNthNonSquare ( n ) ) ; NEW_LINE
def seiresSum ( n , a ) : NEW_LINE INDENT return ( n * ( a [ 0 ] * a [ 0 ] - a [ 2 * n - 1 ] * a [ 2 * n - 1 ] ) / ( 2 * n - 1 ) ) NEW_LINE DEDENT
n = 2 NEW_LINE a = [ 1 , 2 , 3 , 4 ] NEW_LINE print ( int ( seiresSum ( n , a ) ) ) NEW_LINE
def checkdigit ( n , k ) : NEW_LINE INDENT while ( n ) : NEW_LINE DEDENT
rem = n % 10 NEW_LINE
if ( rem == k ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT n = n / 10 NEW_LINE return 0 NEW_LINE
def findNthNumber ( n , k ) : NEW_LINE
i = k + 1 NEW_LINE count = 1 NEW_LINE while ( count < n ) : NEW_LINE
if ( checkdigit ( i , k ) or ( i % k == 0 ) ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT if ( count == n ) : NEW_LINE INDENT return i NEW_LINE DEDENT i += 1 NEW_LINE return - 1 NEW_LINE
n = 10 NEW_LINE k = 2 NEW_LINE print ( findNthNumber ( n , k ) ) NEW_LINE
def find_permutations ( arr ) : NEW_LINE INDENT cnt = 0 NEW_LINE max_ind = - 1 NEW_LINE min_ind = 10000000 ; NEW_LINE n = len ( arr ) NEW_LINE index_of = { } NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT index_of [ arr [ i ] ] = i + 1 NEW_LINE DEDENT for i in range ( 1 , n + 1 ) : NEW_LINE
max_ind = max ( max_ind , index_of [ i ] ) NEW_LINE min_ind = min ( min_ind , index_of [ i ] ) NEW_LINE if ( max_ind - min_ind + 1 == i ) : NEW_LINE INDENT cnt += 1 NEW_LINE DEDENT return cnt NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT nums = [ ] NEW_LINE nums . append ( 2 ) NEW_LINE nums . append ( 3 ) NEW_LINE nums . append ( 1 ) NEW_LINE nums . append ( 5 ) NEW_LINE nums . append ( 4 ) NEW_LINE print ( find_permutations ( nums ) ) NEW_LINE DEDENT
from math import gcd as __gcd NEW_LINE def getCount ( a , n ) : NEW_LINE
gcd = 0 NEW_LINE for i in range ( n ) : NEW_LINE INDENT gcd = __gcd ( gcd , a [ i ] ) NEW_LINE DEDENT
cnt = 0 NEW_LINE for i in range ( 1 , gcd + 1 ) : NEW_LINE INDENT if i * i > gcd : NEW_LINE INDENT break NEW_LINE DEDENT if ( gcd % i == 0 ) : NEW_LINE DEDENT
if ( i * i == gcd ) : NEW_LINE INDENT cnt += 1 NEW_LINE DEDENT
else : NEW_LINE INDENT cnt += 2 NEW_LINE DEDENT return cnt NEW_LINE
a = [ 4 , 16 , 1024 , 48 ] NEW_LINE n = len ( a ) NEW_LINE print ( getCount ( a , n ) ) NEW_LINE
def delCost ( s , cost ) : NEW_LINE
visited = [ False ] * len ( s ) NEW_LINE
ans = 0 NEW_LINE
for i in range ( len ( s ) ) : NEW_LINE
if visited [ i ] : NEW_LINE INDENT continue NEW_LINE DEDENT
maxDel = 0 NEW_LINE
totCost = 0 NEW_LINE
visited [ i ] = True NEW_LINE
for j in range ( i , len ( s ) ) : NEW_LINE
if s [ i ] == s [ j ] : NEW_LINE
maxDel = max ( maxDel , cost [ j ] ) NEW_LINE totCost += cost [ j ] NEW_LINE
visited [ j ] = True NEW_LINE
ans += totCost - maxDel NEW_LINE
return ans NEW_LINE
string = " AAABBB " NEW_LINE
cost = [ 1 , 2 , 3 , 4 , 5 , 6 ] NEW_LINE
string = " AAABBB " NEW_LINE
print ( delCost ( string , cost ) ) NEW_LINE
def checkXOR ( arr , N ) : NEW_LINE
if ( N % 2 == 0 ) : NEW_LINE
xro = 0 ; NEW_LINE
for i in range ( N ) : NEW_LINE
xro ^= arr [ i ] ; NEW_LINE
if ( xro != 0 ) : NEW_LINE INDENT print ( - 1 ) ; NEW_LINE return ; NEW_LINE DEDENT
for i in range ( 0 , N - 3 , 2 ) : NEW_LINE INDENT print ( i , " ▁ " , ( i + 1 ) , " ▁ " , ( i + 2 ) , end = " ▁ " ) ; NEW_LINE DEDENT
for i in range ( 0 , N - 3 , 2 ) : NEW_LINE INDENT print ( i , " ▁ " , ( i + 1 ) , " ▁ " , ( N - 1 ) , end = " ▁ " ) ; NEW_LINE DEDENT else : NEW_LINE
for i in range ( 0 , N - 2 , 2 ) : NEW_LINE INDENT print ( i , " ▁ " , ( i + 1 ) , " ▁ " , ( i + 2 ) ) ; NEW_LINE DEDENT
for i in range ( 0 , N - 2 , 2 ) : NEW_LINE INDENT print ( i , " ▁ " , ( i + 1 ) , " ▁ " , ( N - 1 ) ) ; NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE
arr = [ 4 , 2 , 1 , 7 , 2 ] ; NEW_LINE
N = len ( arr ) ; NEW_LINE
checkXOR ( arr , N ) ; NEW_LINE
def make_array_element_even ( arr , N ) : NEW_LINE
res = 0 NEW_LINE
odd_cont_seg = 0 NEW_LINE
for i in range ( 0 , N ) : NEW_LINE
if ( arr [ i ] % 2 == 1 ) : NEW_LINE
odd_cont_seg += 1 NEW_LINE else : NEW_LINE if ( odd_cont_seg > 0 ) : NEW_LINE
if ( odd_cont_seg % 2 == 0 ) : NEW_LINE
res += odd_cont_seg // 2 NEW_LINE else : NEW_LINE
res += ( odd_cont_seg // 2 ) + 2 NEW_LINE
odd_cont_seg = 0 NEW_LINE
if ( odd_cont_seg > 0 ) : NEW_LINE
if ( odd_cont_seg % 2 == 0 ) : NEW_LINE
res += odd_cont_seg // 2 NEW_LINE else : NEW_LINE
res += odd_cont_seg // 2 + 2 NEW_LINE
return res NEW_LINE
arr = [ 2 , 4 , 5 , 11 , 6 ] NEW_LINE N = len ( arr ) NEW_LINE print ( make_array_element_even ( arr , N ) ) NEW_LINE
def zvalue ( nums ) : NEW_LINE
m = max ( nums ) NEW_LINE cnt = 0 NEW_LINE
for i in range ( 0 , m + 1 , 1 ) : NEW_LINE INDENT cnt = 0 NEW_LINE DEDENT
for j in range ( 0 , len ( nums ) , 1 ) : NEW_LINE
if ( nums [ j ] >= i ) : NEW_LINE INDENT cnt += 1 NEW_LINE DEDENT
if ( cnt == i ) : NEW_LINE INDENT return i NEW_LINE DEDENT
return - 1 NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT nums = [ 7 , 8 , 9 , 0 , 0 , 1 ] NEW_LINE print ( zvalue ( nums ) ) NEW_LINE DEDENT
def lexico_smallest ( s1 , s2 ) : NEW_LINE
M = { } NEW_LINE S = [ ] NEW_LINE pr = { } NEW_LINE
for i in range ( len ( s1 ) ) : NEW_LINE
if s1 [ i ] not in M : NEW_LINE INDENT M [ s1 [ i ] ] = 1 NEW_LINE DEDENT else : NEW_LINE INDENT M [ s1 [ i ] ] += 1 NEW_LINE DEDENT
S . append ( s1 [ i ] ) NEW_LINE S = list ( set ( S ) ) NEW_LINE S . sort ( ) NEW_LINE
for i in range ( len ( s2 ) ) : NEW_LINE INDENT if s2 [ i ] in M : NEW_LINE INDENT M [ s2 [ i ] ] -= 1 NEW_LINE DEDENT DEDENT c = s2 [ 0 ] NEW_LINE index = 0 NEW_LINE res = " " NEW_LINE
for x in S : NEW_LINE
if ( x != c ) : NEW_LINE INDENT for i in range ( 1 , M [ x ] + 1 ) : NEW_LINE INDENT res += x NEW_LINE DEDENT DEDENT else : NEW_LINE
j = 0 NEW_LINE index = len ( res ) NEW_LINE
while ( s2 [ j ] == x ) : NEW_LINE INDENT j += 1 NEW_LINE DEDENT
if ( s2 [ j ] < c ) : NEW_LINE INDENT res += s2 NEW_LINE for i in range ( 1 , M [ x ] + 1 ) : NEW_LINE INDENT res += x NEW_LINE DEDENT DEDENT else : NEW_LINE INDENT for i in range ( 1 , M [ x ] + 1 ) : NEW_LINE INDENT res += x NEW_LINE DEDENT index += M [ x ] NEW_LINE res += s2 NEW_LINE DEDENT pr [ res ] = index NEW_LINE
return pr NEW_LINE
def lexico_largest ( s1 , s2 ) : NEW_LINE
Pr = dict ( lexico_smallest ( s1 , s2 ) ) NEW_LINE
d1 = " " NEW_LINE key = [ * Pr ] [ 0 ] NEW_LINE for i in range ( Pr . get ( key ) - 1 , - 1 , - 1 ) : NEW_LINE INDENT d1 += key [ i ] NEW_LINE DEDENT
d2 = " " NEW_LINE for i in range ( len ( key ) - 1 , Pr [ key ] + len ( s2 ) - 1 , - 1 ) : NEW_LINE INDENT d2 += key [ i ] NEW_LINE DEDENT res = d2 + s2 + d1 NEW_LINE
return res NEW_LINE
s1 = " ethgakagmenpgs " NEW_LINE s2 = " geeks " NEW_LINE
print ( * lexico_smallest ( s1 , s2 ) ) NEW_LINE print ( lexico_largest ( s1 , s2 ) ) NEW_LINE
sz = 100000 NEW_LINE
tree = [ [ ] for i in range ( sz ) ] NEW_LINE
n = 0 NEW_LINE
vis = [ False ] * sz NEW_LINE
subtreeSize = [ 0 for i in range ( sz ) ] NEW_LINE
def addEdge ( a , b ) : NEW_LINE INDENT global tree NEW_LINE DEDENT
tree [ a ] . append ( b ) NEW_LINE
tree [ b ] . append ( a ) NEW_LINE
def dfs ( x ) : NEW_LINE
global vis NEW_LINE global subtreeSize NEW_LINE global tree NEW_LINE vis [ x ] = True NEW_LINE
subtreeSize [ x ] = 1 NEW_LINE
for i in tree [ x ] : NEW_LINE INDENT if ( vis [ i ] == False ) : NEW_LINE INDENT dfs ( i ) NEW_LINE subtreeSize [ x ] += subtreeSize [ i ] NEW_LINE DEDENT DEDENT
def countPairs ( a , b ) : NEW_LINE INDENT global subtreeSize NEW_LINE sub = min ( subtreeSize [ a ] , subtreeSize [ b ] ) NEW_LINE print ( sub * ( n - sub ) ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE
n = 6 NEW_LINE addEdge ( 0 , 1 ) NEW_LINE addEdge ( 0 , 2 ) NEW_LINE addEdge ( 1 , 3 ) NEW_LINE addEdge ( 3 , 4 ) NEW_LINE addEdge ( 3 , 5 ) NEW_LINE
dfs ( 0 ) NEW_LINE
countPairs ( 1 , 3 ) NEW_LINE countPairs ( 0 , 2 ) NEW_LINE
def findPermutation ( arr , N ) : NEW_LINE INDENT pos = len ( arr ) + 1 NEW_LINE DEDENT
if ( pos > N ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT res = 0 NEW_LINE for i in range ( 1 , N + 1 ) : NEW_LINE
if ( i not in arr ) : NEW_LINE
if ( i % pos == 0 or pos % i == 0 ) : NEW_LINE
arr . add ( i ) NEW_LINE
res += findPermutation ( arr , N ) NEW_LINE
arr . remove ( i ) NEW_LINE
return res NEW_LINE
N = 5 NEW_LINE arr = set ( ) NEW_LINE print ( findPermutation ( arr , N ) ) NEW_LINE
def solve ( arr , n , X , Y ) : NEW_LINE
diff = Y - X NEW_LINE
for i in range ( n ) : NEW_LINE INDENT if ( arr [ i ] != 1 ) : NEW_LINE INDENT diff = diff % ( arr [ i ] - 1 ) NEW_LINE DEDENT DEDENT
if ( diff == 0 ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
arr = [ 1 , 2 , 7 , 9 , 10 ] NEW_LINE n = len ( arr ) NEW_LINE X , Y = 11 , 13 NEW_LINE solve ( arr , n , X , Y ) NEW_LINE
maxN = 100001 NEW_LINE
adj = [ [ ] for i in range ( maxN ) ] NEW_LINE
height = [ 0 for i in range ( maxN ) ] NEW_LINE
dist = [ 0 for i in range ( maxN ) ] NEW_LINE
def addEdge ( u , v ) : NEW_LINE
adj [ u ] . append ( v ) NEW_LINE
adj [ v ] . append ( u ) NEW_LINE
def dfs1 ( cur , par ) : NEW_LINE
for u in adj [ cur ] : NEW_LINE INDENT if ( u != par ) : NEW_LINE DEDENT
dfs1 ( u , cur ) NEW_LINE
height [ cur ] = max ( height [ cur ] , height [ u ] ) NEW_LINE
height [ cur ] += 1 NEW_LINE
def dfs2 ( cur , par ) : NEW_LINE INDENT max1 = 0 NEW_LINE max2 = 0 NEW_LINE DEDENT
for u in adj [ cur ] : NEW_LINE INDENT if ( u != par ) : NEW_LINE DEDENT
if ( height [ u ] >= max1 ) : NEW_LINE INDENT max2 = max1 NEW_LINE max1 = height [ u ] NEW_LINE DEDENT elif ( height [ u ] > max2 ) : NEW_LINE INDENT max2 = height [ u ] NEW_LINE DEDENT sum = 0 NEW_LINE for u in adj [ cur ] : NEW_LINE if ( u != par ) : NEW_LINE
sum = ( max2 if ( max1 == height [ u ] ) else max1 ) NEW_LINE if ( max1 == height [ u ] ) : NEW_LINE INDENT dist [ u ] = 1 + max ( 1 + max2 , dist [ cur ] ) NEW_LINE DEDENT else : NEW_LINE INDENT dist [ u ] = 1 + max ( 1 + max1 , dist [ cur ] ) NEW_LINE DEDENT
dfs2 ( u , cur ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 6 NEW_LINE addEdge ( 1 , 2 ) NEW_LINE addEdge ( 2 , 3 ) NEW_LINE addEdge ( 2 , 4 ) NEW_LINE addEdge ( 2 , 5 ) NEW_LINE addEdge ( 5 , 6 ) NEW_LINE DEDENT
dfs1 ( 1 , 0 ) NEW_LINE
dfs2 ( 1 , 0 ) NEW_LINE
for i in range ( 1 , n + 1 ) : NEW_LINE INDENT print ( max ( dist [ i ] , height [ i ] ) - 1 , end = ' ▁ ' ) NEW_LINE DEDENT
def middleOfThree ( a , b , c ) : NEW_LINE
INDENT def middleOfThree ( a , b , c ) : NEW_LINE DEDENT
if ( ( a < b and b < c ) or ( c < b and b < a ) ) : NEW_LINE INDENT return b ; NEW_LINE DEDENT
if ( ( b < a and a < c ) or ( c < a and a < b ) ) : NEW_LINE INDENT return a ; NEW_LINE DEDENT else : NEW_LINE INDENT return c NEW_LINE DEDENT
a = 20 NEW_LINE b = 30 NEW_LINE c = 40 NEW_LINE print ( middleOfThree ( a , b , c ) ) NEW_LINE
def selectionSort ( arr , n ) : NEW_LINE
for i in range ( n - 1 ) : NEW_LINE
min_idx = i NEW_LINE for j in range ( i + 1 , n ) : NEW_LINE INDENT if ( arr [ j ] < arr [ min_idx ] ) : NEW_LINE INDENT min_idx = j NEW_LINE DEDENT DEDENT
arr [ min_idx ] , arr [ i ] = arr [ i ] , arr [ min_idx ] NEW_LINE
def printArray ( arr , size ) : NEW_LINE INDENT for i in range ( size ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT print ( ) NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 64 , 25 , 12 , 22 , 11 ] NEW_LINE n = len ( arr ) NEW_LINE DEDENT
selectionSort ( arr , n ) NEW_LINE print ( " Sorted ▁ array : ▁ " ) NEW_LINE
printArray ( arr , n ) NEW_LINE
def checkStr1CanConStr2 ( str1 , str2 ) : NEW_LINE
N = len ( str1 ) NEW_LINE
M = len ( str2 ) NEW_LINE
st1 = set ( [ ] ) NEW_LINE
st2 = set ( [ ] ) NEW_LINE
hash1 = [ 0 ] * 256 NEW_LINE
for i in range ( N ) : NEW_LINE
hash1 [ ord ( str1 [ i ] ) ] += 1 NEW_LINE
for i in range ( N ) : NEW_LINE
st1 . add ( str1 [ i ] ) NEW_LINE
for i in range ( M ) : NEW_LINE
st2 . add ( str2 [ i ] ) NEW_LINE
if ( st1 != st2 ) : NEW_LINE INDENT return False NEW_LINE DEDENT
hash2 = [ 0 ] * 256 NEW_LINE
for i in range ( M ) : NEW_LINE
hash2 [ ord ( str2 [ i ] ) ] += 1 NEW_LINE
hash1 . sort ( ) NEW_LINE
hash2 . sort ( ) NEW_LINE
for i in range ( 256 ) : NEW_LINE
if ( hash1 [ i ] != hash2 [ i ] ) : NEW_LINE INDENT return False NEW_LINE DEDENT return True NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT str1 = " xyyzzlll " NEW_LINE str2 = " yllzzxxx " NEW_LINE if ( checkStr1CanConStr2 ( str1 , str2 ) ) : NEW_LINE INDENT print ( " True " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " False " ) NEW_LINE DEDENT DEDENT
def partSort ( arr , N , a , b ) : NEW_LINE
l = min ( a , b ) NEW_LINE r = max ( a , b ) NEW_LINE arr = ( arr [ 0 : l ] + sorted ( arr [ l : r + 1 ] ) + arr [ r : N ] ) NEW_LINE
for i in range ( 0 , N , 1 ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 7 , 8 , 4 , 5 , 2 ] NEW_LINE a = 1 NEW_LINE b = 4 NEW_LINE N = len ( arr ) NEW_LINE partSort ( arr , N , a , b ) NEW_LINE DEDENT
INF = 2147483647 NEW_LINE N = 4 NEW_LINE
def minCost ( cost ) : NEW_LINE
dist = [ 0 for i in range ( N ) ] NEW_LINE for i in range ( N ) : NEW_LINE INDENT dist [ i ] = INF NEW_LINE DEDENT dist [ 0 ] = 0 NEW_LINE
for i in range ( N ) : NEW_LINE INDENT for j in range ( i + 1 , N ) : NEW_LINE INDENT if ( dist [ j ] > dist [ i ] + cost [ i ] [ j ] ) : NEW_LINE INDENT dist [ j ] = dist [ i ] + cost [ i ] [ j ] NEW_LINE DEDENT DEDENT DEDENT return dist [ N - 1 ] NEW_LINE
cost = [ [ 0 , 15 , 80 , 90 ] , [ INF , 0 , 40 , 50 ] , [ INF , INF , 0 , 70 ] , [ INF , INF , INF , 0 ] ] NEW_LINE print ( " The ▁ Minimum ▁ cost ▁ to ▁ reach ▁ station ▁ " , N , " ▁ is ▁ " , minCost ( cost ) ) NEW_LINE
def numOfways ( n , k ) : NEW_LINE INDENT p = 1 NEW_LINE if ( k % 2 ) : NEW_LINE INDENT p = - 1 NEW_LINE DEDENT return ( pow ( n - 1 , k ) + p * ( n - 1 ) ) / n NEW_LINE DEDENT
n = 4 NEW_LINE k = 2 NEW_LINE print ( numOfways ( n , k ) ) NEW_LINE
def largest_alphabet ( a , n ) : NEW_LINE
' NEW_LINE INDENT max = ' A ' NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT if ( a [ i ] > max ) : NEW_LINE INDENT max = a [ i ] NEW_LINE DEDENT DEDENT
return max NEW_LINE
def smallest_alphabet ( a , n ) : NEW_LINE
' NEW_LINE INDENT min = ' z ' ; NEW_LINE DEDENT
for i in range ( n - 1 ) : NEW_LINE INDENT if ( a [ i ] < min ) : NEW_LINE INDENT min = a [ i ] NEW_LINE DEDENT DEDENT
return min NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
a = " GeEksforGeeks " NEW_LINE
size = len ( a ) NEW_LINE
print ( " Largest ▁ and ▁ smallest ▁ alphabet ▁ is ▁ : ▁ " , end = " " ) NEW_LINE print ( largest_alphabet ( a , size ) , end = " ▁ and ▁ " ) NEW_LINE print ( smallest_alphabet ( a , size ) ) NEW_LINE ' NEW_LINE
def maximumPalinUsingKChanges ( strr , k ) : NEW_LINE INDENT palin = strr [ : : ] NEW_LINE DEDENT
l = 0 NEW_LINE r = len ( strr ) - 1 NEW_LINE
while ( l <= r ) : NEW_LINE
if ( strr [ l ] != strr [ r ] ) : NEW_LINE INDENT palin [ l ] = palin [ r ] = NEW_LINE INDENT max ( strr [ l ] , strr [ r ] ) NEW_LINE DEDENT k -= 1 NEW_LINE DEDENT l += 1 NEW_LINE r -= 1 NEW_LINE
if ( k < 0 ) : NEW_LINE INDENT return " Not ▁ possible " NEW_LINE DEDENT l = 0 NEW_LINE r = len ( strr ) - 1 NEW_LINE while ( l <= r ) : NEW_LINE
if ( l == r ) : NEW_LINE INDENT if ( k > 0 ) : NEW_LINE INDENT palin [ l ] = '9' NEW_LINE DEDENT DEDENT
if ( palin [ l ] < '9' ) : NEW_LINE
if ( k >= 2 and palin [ l ] == strr [ l ] and palin [ r ] == strr [ r ] ) : NEW_LINE INDENT k -= 2 NEW_LINE palin [ l ] = palin [ r ] = '9' NEW_LINE DEDENT
elif ( k >= 1 and ( palin [ l ] != strr [ l ] or palin [ r ] != strr [ r ] ) ) : NEW_LINE INDENT k -= 1 NEW_LINE palin [ l ] = palin [ r ] = '9' NEW_LINE DEDENT l += 1 NEW_LINE r -= 1 NEW_LINE return palin NEW_LINE
st = "43435" NEW_LINE strr = [ i for i in st ] NEW_LINE k = 3 NEW_LINE a = maximumPalinUsingKChanges ( strr , k ) NEW_LINE print ( " " . join ( a ) ) NEW_LINE
def countTriplets ( A ) : NEW_LINE
cnt = 0 ; NEW_LINE
tuples = { } ; NEW_LINE
for a in A : NEW_LINE
for b in A : NEW_LINE INDENT if ( a & b ) in tuples : NEW_LINE INDENT tuples [ a & b ] += 1 ; NEW_LINE DEDENT else : NEW_LINE INDENT tuples [ a & b ] = 1 ; NEW_LINE DEDENT DEDENT
for a in A : NEW_LINE
for t in tuples : NEW_LINE
if ( ( t & a ) == 0 ) : NEW_LINE INDENT cnt += tuples [ t ] ; NEW_LINE DEDENT
return cnt ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
A = [ 2 , 1 , 3 ] ; NEW_LINE
print ( countTriplets ( A ) ) ; NEW_LINE
mn = 1000 NEW_LINE
def parity ( even , odd , v , i ) : NEW_LINE INDENT global mn NEW_LINE DEDENT
if ( i == len ( v ) or len ( even ) == 0 or len ( odd ) == 0 ) : NEW_LINE INDENT count = 0 NEW_LINE for j in range ( len ( v ) - 1 ) : NEW_LINE INDENT if ( v [ j ] % 2 != v [ j + 1 ] % 2 ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT DEDENT if ( count < mn ) : NEW_LINE INDENT mn = count NEW_LINE DEDENT return NEW_LINE DEDENT
if ( v [ i ] != - 1 ) : NEW_LINE INDENT parity ( even , odd , v , i + 1 ) NEW_LINE DEDENT
else : NEW_LINE INDENT if ( len ( even ) != 0 ) : NEW_LINE INDENT x = even [ len ( even ) - 1 ] NEW_LINE even . remove ( even [ len ( even ) - 1 ] ) NEW_LINE v [ i ] = x NEW_LINE parity ( even , odd , v , i + 1 ) NEW_LINE DEDENT DEDENT
even . append ( x ) NEW_LINE if ( len ( odd ) != 0 ) : NEW_LINE x = odd [ len ( odd ) - 1 ] NEW_LINE odd . remove ( odd [ len ( odd ) - 1 ] ) NEW_LINE v [ i ] = x NEW_LINE parity ( even , odd , v , i + 1 ) NEW_LINE
odd . append ( x ) NEW_LINE
def mnDiffParity ( v , n ) : NEW_LINE INDENT global mn NEW_LINE DEDENT
even = [ ] NEW_LINE
odd = [ ] NEW_LINE m = { i : 0 for i in range ( 100 ) } NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE INDENT m [ i ] = 1 NEW_LINE DEDENT for i in range ( len ( v ) ) : NEW_LINE
if ( v [ i ] != - 1 ) : NEW_LINE INDENT m . pop ( v [ i ] ) NEW_LINE DEDENT
for key in m . keys ( ) : NEW_LINE INDENT if ( key % 2 == 0 ) : NEW_LINE INDENT even . append ( key ) NEW_LINE DEDENT else : NEW_LINE INDENT odd . append ( key ) NEW_LINE DEDENT DEDENT parity ( even , odd , v , 0 ) NEW_LINE print ( mn + 4 ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 8 NEW_LINE v = [ 2 , 1 , 4 , - 1 , - 1 , 6 , - 1 , 8 ] NEW_LINE mnDiffParity ( v , n ) NEW_LINE DEDENT
MAX = 100005 NEW_LINE adjacent = [ [ ] for i in range ( MAX ) ] NEW_LINE visited = [ False ] * MAX NEW_LINE
startnode = endnode = thirdnode = None NEW_LINE maxi , N = - 1 , None NEW_LINE
parent = [ None ] * MAX NEW_LINE
vis = [ False ] * MAX NEW_LINE
def dfs ( u , count ) : NEW_LINE INDENT visited [ u ] = True NEW_LINE temp = 0 NEW_LINE global startnode , maxi NEW_LINE for i in range ( 0 , len ( adjacent [ u ] ) ) : NEW_LINE INDENT if not visited [ adjacent [ u ] [ i ] ] : NEW_LINE INDENT temp += 1 NEW_LINE dfs ( adjacent [ u ] [ i ] , count + 1 ) NEW_LINE DEDENT DEDENT if temp == 0 : NEW_LINE INDENT if maxi < count : NEW_LINE INDENT maxi = count NEW_LINE startnode = u NEW_LINE DEDENT DEDENT DEDENT
def dfs1 ( u , count ) : NEW_LINE INDENT visited [ u ] = True NEW_LINE temp = 0 NEW_LINE global endnode , maxi NEW_LINE for i in range ( 0 , len ( adjacent [ u ] ) ) : NEW_LINE INDENT if not visited [ adjacent [ u ] [ i ] ] : NEW_LINE INDENT temp += 1 NEW_LINE parent [ adjacent [ u ] [ i ] ] = u NEW_LINE dfs1 ( adjacent [ u ] [ i ] , count + 1 ) NEW_LINE DEDENT DEDENT if temp == 0 : NEW_LINE INDENT if maxi < count : NEW_LINE INDENT maxi = count NEW_LINE endnode = u NEW_LINE DEDENT DEDENT DEDENT
def dfs2 ( u , count ) : NEW_LINE INDENT visited [ u ] = True NEW_LINE temp = 0 NEW_LINE global thirdnode , maxi NEW_LINE for i in range ( 0 , len ( adjacent [ u ] ) ) : NEW_LINE INDENT if ( not visited [ adjacent [ u ] [ i ] ] and not vis [ adjacent [ u ] [ i ] ] ) : NEW_LINE INDENT temp += 1 NEW_LINE dfs2 ( adjacent [ u ] [ i ] , count + 1 ) NEW_LINE DEDENT DEDENT if temp == 0 : NEW_LINE INDENT if maxi < count : NEW_LINE INDENT maxi = count NEW_LINE thirdnode = u NEW_LINE DEDENT DEDENT DEDENT
def findNodes ( ) : NEW_LINE
dfs ( 1 , 0 ) NEW_LINE global maxi NEW_LINE for i in range ( 0 , N + 1 ) : NEW_LINE INDENT visited [ i ] = False NEW_LINE DEDENT maxi = - 1 NEW_LINE
dfs1 ( startnode , 0 ) NEW_LINE for i in range ( 0 , N + 1 ) : NEW_LINE INDENT visited [ i ] = False NEW_LINE DEDENT
x = endnode NEW_LINE vis [ startnode ] = True NEW_LINE
while x != startnode : NEW_LINE INDENT vis [ x ] = True NEW_LINE x = parent [ x ] NEW_LINE DEDENT maxi = - 1 NEW_LINE
for i in range ( 1 , N + 1 ) : NEW_LINE INDENT if vis [ i ] : NEW_LINE INDENT dfs2 ( i , 0 ) NEW_LINE DEDENT DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT N = 4 NEW_LINE adjacent [ 1 ] . append ( 2 ) NEW_LINE adjacent [ 2 ] . append ( 1 ) NEW_LINE adjacent [ 1 ] . append ( 3 ) NEW_LINE adjacent [ 3 ] . append ( 1 ) NEW_LINE adjacent [ 1 ] . append ( 4 ) NEW_LINE adjacent [ 4 ] . append ( 1 ) NEW_LINE findNodes ( ) NEW_LINE print ( " ( { } , ▁ { } , ▁ { } ) " . format ( startnode , endnode , thirdnode ) ) NEW_LINE DEDENT
def newvol ( x ) : NEW_LINE INDENT print ( " percentage ▁ increase ▁ in ▁ the " , pow ( x , 3 ) / 10000 + 3 * x + ( 3 * pow ( x , 2 ) ) / 100 , " % " ) DEDENT
x = 10.0 NEW_LINE newvol ( x ) NEW_LINE
import math as mt NEW_LINE
def length_of_chord ( r , x ) : NEW_LINE INDENT print ( " The ▁ length ▁ of ▁ the ▁ chord " , " ▁ of ▁ the ▁ circle ▁ is ▁ " , 2 * r * mt . sin ( x * ( 3.14 / 180 ) ) ) NEW_LINE DEDENT
r = 4 NEW_LINE x = 63 ; NEW_LINE length_of_chord ( r , x ) NEW_LINE
from math import * NEW_LINE
def area ( a ) : NEW_LINE
if a < 0 : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
area = sqrt ( a ) / 6 NEW_LINE return area NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = 10 NEW_LINE print ( round ( area ( a ) , 6 ) ) NEW_LINE DEDENT
from math import * NEW_LINE
def longestRodInCuboid ( length , breadth , height ) : NEW_LINE
temp = length * length + breadth * breadth + height * height NEW_LINE
result = sqrt ( temp ) NEW_LINE return result NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT length , breadth , height = 12 , 9 , 8 NEW_LINE DEDENT
print ( longestRodInCuboid ( length , breadth , height ) ) NEW_LINE
def LiesInsieRectangle ( a , b , x , y ) : NEW_LINE INDENT if ( x - y - b <= 0 and x - y + b >= 0 and x + y - 2 * a + b <= 0 and x + y - b >= 0 ) : NEW_LINE INDENT return True NEW_LINE DEDENT return False NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a , b , x , y = 7 , 2 , 4 , 5 NEW_LINE if LiesInsieRectangle ( a , b , x , y ) : NEW_LINE INDENT print ( " Given ▁ point ▁ lies ▁ inside " " ▁ the ▁ rectangle " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Given ▁ point ▁ does ▁ not ▁ lie " " ▁ on ▁ the ▁ rectangle " ) NEW_LINE DEDENT DEDENT
def maxvolume ( s ) : NEW_LINE INDENT maxvalue = 0 NEW_LINE DEDENT
i = 1 NEW_LINE for i in range ( s - 1 ) : NEW_LINE INDENT j = 1 NEW_LINE DEDENT
for j in range ( s ) : NEW_LINE
k = s - i - j NEW_LINE
maxvalue = max ( maxvalue , i * j * k ) NEW_LINE return maxvalue NEW_LINE
s = 8 NEW_LINE print ( maxvolume ( s ) ) NEW_LINE
def maxvolume ( s ) : NEW_LINE
length = int ( s / 3 ) NEW_LINE s -= length NEW_LINE
breadth = s / 2 NEW_LINE
height = s - breadth NEW_LINE return int ( length * breadth * height ) NEW_LINE
s = 8 NEW_LINE print ( maxvolume ( s ) ) NEW_LINE
import math NEW_LINE
def hexagonArea ( s ) : NEW_LINE INDENT return ( ( 3 * math . sqrt ( 3 ) * ( s * s ) ) / 2 ) ; NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE
s = 4 NEW_LINE print ( " Area : " , " { 0 : . 4f } " . format ( hexagonArea ( s ) ) ) NEW_LINE
def maxSquare ( b , m ) : NEW_LINE
return ( b / m - 1 ) * ( b / m ) / 2 NEW_LINE
b = 10 NEW_LINE m = 2 NEW_LINE print ( int ( maxSquare ( b , m ) ) ) NEW_LINE
from math import sqrt NEW_LINE
def findRightAngle ( A , H ) : NEW_LINE
D = pow ( H , 4 ) - 16 * A * A NEW_LINE if D >= 0 : NEW_LINE
root1 = ( H * H + sqrt ( D ) ) / 2 NEW_LINE root2 = ( H * H - sqrt ( D ) ) / 2 NEW_LINE a = sqrt ( root1 ) NEW_LINE b = sqrt ( root2 ) NEW_LINE if b >= a : NEW_LINE INDENT print a , b , H NEW_LINE DEDENT else : NEW_LINE INDENT print b , a , H NEW_LINE DEDENT else : NEW_LINE print " - 1" NEW_LINE
findRightAngle ( 6 , 5 ) NEW_LINE
def numberOfSquares ( base ) : NEW_LINE
base = ( base - 2 ) NEW_LINE
base = base // 2 NEW_LINE return base * ( base + 1 ) / 2 NEW_LINE
base = 8 NEW_LINE print ( numberOfSquares ( base ) ) NEW_LINE
def performQuery ( arr , Q ) : NEW_LINE
for i in range ( 0 , len ( Q ) ) : NEW_LINE
orr = 0 NEW_LINE
x = Q [ i ] [ 0 ] NEW_LINE arr [ x - 1 ] = Q [ i ] [ 1 ] NEW_LINE
for j in range ( 0 , len ( arr ) ) : NEW_LINE INDENT orr = orr | arr [ j ] NEW_LINE DEDENT
print ( orr , end = " ▁ " ) NEW_LINE
arr = [ 1 , 2 , 3 ] NEW_LINE Q = [ [ 1 , 4 ] , [ 3 , 0 ] ] NEW_LINE performQuery ( arr , Q ) NEW_LINE
def smallest ( k , d ) : NEW_LINE INDENT cnt = 1 NEW_LINE m = d % k NEW_LINE DEDENT
v = [ 0 for i in range ( k ) ] ; NEW_LINE v [ m ] = 1 NEW_LINE
while ( 1 ) : NEW_LINE INDENT if ( m == 0 ) : NEW_LINE INDENT return cnt NEW_LINE DEDENT m = ( ( ( m * ( 10 % k ) ) % k ) + ( d % k ) ) % k NEW_LINE DEDENT
if ( v [ m ] == 1 ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT v [ m ] = 1 NEW_LINE cnt += 1 NEW_LINE return - 1 NEW_LINE
d = 1 NEW_LINE k = 41 NEW_LINE print ( smallest ( k , d ) ) NEW_LINE
def fib ( n ) : NEW_LINE INDENT if n <= 1 : NEW_LINE INDENT return n NEW_LINE DEDENT return fib ( n - 1 ) + fib ( n - 2 ) NEW_LINE DEDENT
def findVertices ( n ) : NEW_LINE
return fib ( n + 2 ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 3 NEW_LINE print ( findVertices ( n ) ) NEW_LINE DEDENT
import math NEW_LINE
def checkCommonDivisor ( arr , N , X ) : NEW_LINE
G = 0 NEW_LINE
for i in range ( N ) : NEW_LINE INDENT G = math . gcd ( G , arr [ i ] ) NEW_LINE DEDENT copy_G = G NEW_LINE for divisor in range ( 2 , X + 1 ) : NEW_LINE
while ( G % divisor == 0 ) : NEW_LINE
G = G // divisor NEW_LINE
if ( G <= X ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT
for i in range ( N ) : NEW_LINE INDENT print ( arr [ i ] // copy_G , end = " ▁ " ) NEW_LINE DEDENT print ( ) NEW_LINE
else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE
arr = [ 6 , 15 , 6 ] NEW_LINE X = 6 NEW_LINE
N = len ( arr ) NEW_LINE checkCommonDivisor ( arr , N , X ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , next = None , prev = None , data = None ) : NEW_LINE INDENT self . next = next NEW_LINE self . prev = prev NEW_LINE self . data = data NEW_LINE DEDENT DEDENT
def reverse ( head_ref ) : NEW_LINE INDENT temp = None NEW_LINE current = head_ref NEW_LINE DEDENT
while ( current != None ) : NEW_LINE INDENT temp = current . prev NEW_LINE current . prev = current . next NEW_LINE current . next = temp NEW_LINE current = current . prev NEW_LINE DEDENT
if ( temp != None ) : NEW_LINE INDENT head_ref = temp . prev NEW_LINE return head_ref NEW_LINE DEDENT
def merge ( first , second ) : NEW_LINE
if ( first == None ) : NEW_LINE INDENT return second NEW_LINE DEDENT
if ( second == None ) : NEW_LINE INDENT return first NEW_LINE DEDENT
if ( first . data < second . data ) : NEW_LINE INDENT first . next = merge ( first . next , second ) NEW_LINE first . next . prev = first NEW_LINE first . prev = None NEW_LINE return first NEW_LINE DEDENT else : NEW_LINE INDENT second . next = merge ( first , second . next ) NEW_LINE second . next . prev = second NEW_LINE second . prev = None NEW_LINE return second NEW_LINE DEDENT
def sort ( head ) : NEW_LINE
if ( head == None or head . next == None ) : NEW_LINE INDENT return head NEW_LINE DEDENT current = head . next NEW_LINE while ( current != None ) : NEW_LINE
if ( current . data < current . prev . data ) : NEW_LINE INDENT break NEW_LINE DEDENT
current = current . next NEW_LINE
if ( current == None ) : NEW_LINE INDENT return head NEW_LINE DEDENT
current . prev . next = None NEW_LINE current . prev = None NEW_LINE
current = reverse ( current ) NEW_LINE
return merge ( head , current ) NEW_LINE
def push ( head_ref , new_data ) : NEW_LINE
new_node = Node ( ) NEW_LINE
new_node . data = new_data NEW_LINE
new_node . prev = None NEW_LINE
new_node . next = ( head_ref ) NEW_LINE
if ( ( head_ref ) != None ) : NEW_LINE INDENT ( head_ref ) . prev = new_node NEW_LINE DEDENT
( head_ref ) = new_node NEW_LINE return head_ref NEW_LINE
def printList ( head ) : NEW_LINE
if ( head == None ) : NEW_LINE INDENT print ( " Doubly ▁ Linked ▁ list ▁ empty " ) NEW_LINE DEDENT while ( head != None ) : NEW_LINE INDENT print ( head . data , end = " ▁ " ) NEW_LINE head = head . next NEW_LINE DEDENT
head = None NEW_LINE
head = push ( head , 1 ) NEW_LINE head = push ( head , 4 ) NEW_LINE head = push ( head , 6 ) NEW_LINE head = push ( head , 10 ) NEW_LINE head = push ( head , 12 ) NEW_LINE head = push ( head , 7 ) NEW_LINE head = push ( head , 5 ) NEW_LINE head = push ( head , 2 ) NEW_LINE print ( " Original ▁ Doubly ▁ linked ▁ list : n " ) NEW_LINE printList ( head ) NEW_LINE
head = sort ( head ) NEW_LINE print ( " Doubly linked list after sorting : " ) NEW_LINE printList ( head ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , x ) : NEW_LINE INDENT self . data = x NEW_LINE self . next = None NEW_LINE DEDENT DEDENT
def printlist ( head ) : NEW_LINE INDENT if ( not head ) : NEW_LINE INDENT print ( " Empty ▁ List " ) NEW_LINE return NEW_LINE DEDENT while ( head != None ) : NEW_LINE INDENT print ( head . data , end = " ▁ " ) NEW_LINE if ( head . next ) : NEW_LINE INDENT print ( end = " - > ▁ " ) NEW_LINE DEDENT head = head . next NEW_LINE DEDENT print ( ) NEW_LINE DEDENT
def isVowel ( x ) : NEW_LINE INDENT return ( x == ' a ' or x == ' e ' or x == ' i ' or x == ' o ' or x == ' u ' or x == ' A ' or x == ' E ' or x == ' I ' or x == ' O ' or x == ' U ' ) NEW_LINE DEDENT
def arrange ( head ) : NEW_LINE INDENT newHead = head NEW_LINE DEDENT
latestVowel = None NEW_LINE curr = head NEW_LINE
if ( head == None ) : NEW_LINE INDENT return None NEW_LINE DEDENT
if ( isVowel ( head . data ) ) : NEW_LINE
latestVowel = head NEW_LINE else : NEW_LINE
while ( curr . next != None and not isVowel ( curr . next . data ) ) : NEW_LINE INDENT curr = curr . next NEW_LINE DEDENT
if ( curr . next == None ) : NEW_LINE INDENT return head NEW_LINE DEDENT
latestVowel = newHead = curr . next NEW_LINE curr . next = curr . next . next NEW_LINE latestVowel . next = head NEW_LINE
while ( curr != None and curr . next != None ) : NEW_LINE INDENT if ( isVowel ( curr . next . data ) ) : NEW_LINE DEDENT
if ( curr == latestVowel ) : NEW_LINE
latestVowel = curr = curr . next NEW_LINE else : NEW_LINE
temp = latestVowel . next NEW_LINE
latestVowel . next = curr . next NEW_LINE
latestVowel = latestVowel . next NEW_LINE
curr . next = curr . next . next NEW_LINE
latestVowel . next = temp NEW_LINE else : NEW_LINE
curr = curr . next NEW_LINE return newHead NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT head = Node ( ' a ' ) NEW_LINE head . next = Node ( ' b ' ) NEW_LINE head . next . next = Node ( ' c ' ) NEW_LINE head . next . next . next = Node ( ' e ' ) NEW_LINE head . next . next . next . next = Node ( ' d ' ) NEW_LINE head . next . next . next . next . next = Node ( ' o ' ) NEW_LINE head . next . next . next . next . next . next = Node ( ' x ' ) NEW_LINE head . next . next . next . next . next . next . next = Node ( ' i ' ) NEW_LINE print ( " Linked ▁ list ▁ before ▁ : " ) NEW_LINE printlist ( head ) NEW_LINE head = arrange ( head ) NEW_LINE print ( " Linked ▁ list ▁ after ▁ : " ) NEW_LINE printlist ( head ) NEW_LINE DEDENT
class newNode : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . right = self . left = None NEW_LINE DEDENT DEDENT def KthLargestUsingMorrisTraversal ( root , k ) : NEW_LINE INDENT curr = root NEW_LINE Klargest = None NEW_LINE DEDENT
count = 0 NEW_LINE while ( curr != None ) : NEW_LINE
if ( curr . right == None ) : NEW_LINE
count += 1 NEW_LINE if ( count == k ) : NEW_LINE INDENT Klargest = curr NEW_LINE DEDENT
curr = curr . left NEW_LINE else : NEW_LINE
succ = curr . right NEW_LINE while ( succ . left != None and succ . left != curr ) : NEW_LINE INDENT succ = succ . left NEW_LINE DEDENT if ( succ . left == None ) : NEW_LINE
succ . left = curr NEW_LINE
curr = curr . right NEW_LINE
else : NEW_LINE INDENT succ . left = None NEW_LINE count += 1 NEW_LINE if ( count == k ) : NEW_LINE INDENT Klargest = curr NEW_LINE DEDENT DEDENT
curr = curr . left NEW_LINE return Klargest NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
root = newNode ( 4 ) NEW_LINE root . left = newNode ( 2 ) NEW_LINE root . right = newNode ( 7 ) NEW_LINE root . left . left = newNode ( 1 ) NEW_LINE root . left . right = newNode ( 3 ) NEW_LINE root . right . left = newNode ( 6 ) NEW_LINE root . right . right = newNode ( 10 ) NEW_LINE print ( " Finding ▁ K - th ▁ largest ▁ Node ▁ in ▁ BST ▁ : ▁ " , KthLargestUsingMorrisTraversal ( root , 2 ) . data ) NEW_LINE
MAX_SIZE = 10 NEW_LINE
def sortByRow ( mat , n , ascending ) : NEW_LINE INDENT for i in range ( n ) : NEW_LINE INDENT if ( ascending ) : NEW_LINE INDENT mat [ i ] . sort ( ) NEW_LINE DEDENT else : NEW_LINE INDENT mat [ i ] . sort ( reverse = True ) NEW_LINE DEDENT DEDENT DEDENT
def transpose ( mat , n ) : NEW_LINE INDENT for i in range ( n ) : NEW_LINE INDENT for j in range ( i + 1 , n ) : NEW_LINE DEDENT DEDENT
temp = mat [ i ] [ j ] NEW_LINE mat [ i ] [ j ] = mat [ j ] [ i ] NEW_LINE mat [ j ] [ i ] = temp NEW_LINE
def sortMatRowAndColWise ( mat , n ) : NEW_LINE
sortByRow ( mat , n , True ) NEW_LINE
transpose ( mat , n ) NEW_LINE
sortByRow ( mat , n , False ) NEW_LINE
transpose ( mat , n ) NEW_LINE
def printMat ( mat , n ) : NEW_LINE INDENT for i in range ( n ) : NEW_LINE INDENT for j in range ( n ) : NEW_LINE INDENT print ( mat [ i ] [ j ] , " ▁ " , end = " " ) NEW_LINE DEDENT print ( ) NEW_LINE DEDENT DEDENT
n = 3 NEW_LINE mat = [ [ 3 , 2 , 1 ] , [ 9 , 8 , 7 ] , [ 6 , 5 , 4 ] ] NEW_LINE print ( " Original ▁ Matrix : " ) NEW_LINE printMat ( mat , n ) NEW_LINE sortMatRowAndColWise ( mat , n ) NEW_LINE print ( " Matrix ▁ After ▁ Sorting : " ) NEW_LINE printMat ( mat , n ) NEW_LINE
MAX_SIZE = 10 NEW_LINE
def sortByRow ( mat , n ) : NEW_LINE INDENT for i in range ( n ) : NEW_LINE DEDENT
for j in range ( n - 1 ) : NEW_LINE INDENT if mat [ i ] [ j ] > mat [ i ] [ j + 1 ] : NEW_LINE INDENT temp = mat [ i ] [ j ] NEW_LINE mat [ i ] [ j ] = mat [ i ] [ j + 1 ] NEW_LINE mat [ i ] [ j + 1 ] = temp NEW_LINE DEDENT DEDENT
def transpose ( mat , n ) : NEW_LINE INDENT for i in range ( n ) : NEW_LINE INDENT for j in range ( i + 1 , n ) : NEW_LINE DEDENT DEDENT
t = mat [ i ] [ j ] NEW_LINE mat [ i ] [ j ] = mat [ j ] [ i ] NEW_LINE mat [ j ] [ i ] = t NEW_LINE
def sortMatRowAndColWise ( mat , n ) : NEW_LINE
sortByRow ( mat , n ) NEW_LINE
transpose ( mat , n ) NEW_LINE
sortByRow ( mat , n ) NEW_LINE
transpose ( mat , n ) NEW_LINE
def printMat ( mat , n ) : NEW_LINE INDENT for i in range ( n ) : NEW_LINE INDENT for j in range ( n ) : NEW_LINE INDENT print ( str ( mat [ i ] [ j ] ) , end = " ▁ " ) NEW_LINE DEDENT print ( ) ; NEW_LINE DEDENT DEDENT
mat = [ [ 4 , 1 , 3 ] , [ 9 , 6 , 8 ] , [ 5 , 2 , 7 ] ] NEW_LINE n = 3 NEW_LINE print ( " Original ▁ Matrix : " ) NEW_LINE printMat ( mat , n ) NEW_LINE sortMatRowAndColWise ( mat , n ) NEW_LINE print ( " Matrix After Sorting : " ) NEW_LINE printMat ( mat , n ) NEW_LINE
def DoublyEven ( n ) : NEW_LINE
arr = [ [ ( n * y ) + x + 1 for x in range ( n ) ] for y in range ( n ) ] NEW_LINE
for i in range ( 0 , n / 4 ) : NEW_LINE INDENT for j in range ( 0 , n / 4 ) : NEW_LINE INDENT arr [ i ] [ j ] = ( n * n + 1 ) - arr [ i ] [ j ] ; NEW_LINE DEDENT DEDENT
for i in range ( 0 , n / 4 ) : NEW_LINE INDENT for j in range ( 3 * ( n / 4 ) , n ) : NEW_LINE INDENT arr [ i ] [ j ] = ( n * n + 1 ) - arr [ i ] [ j ] ; NEW_LINE DEDENT DEDENT
for i in range ( 3 * ( n / 4 ) , n ) : NEW_LINE INDENT for j in range ( 0 , n / 4 ) : NEW_LINE INDENT arr [ i ] [ j ] = ( n * n + 1 ) - arr [ i ] [ j ] ; NEW_LINE DEDENT DEDENT
for i in range ( 3 * ( n / 4 ) , n ) : NEW_LINE INDENT for j in range ( 3 * ( n / 4 ) , n ) : NEW_LINE INDENT arr [ i ] [ j ] = ( n * n + 1 ) - arr [ i ] [ j ] ; NEW_LINE DEDENT DEDENT
for i in range ( n / 4 , 3 * ( n / 4 ) ) : NEW_LINE INDENT for j in range ( n / 4 , 3 * ( n / 4 ) ) : NEW_LINE INDENT arr [ i ] [ j ] = ( n * n + 1 ) - arr [ i ] [ j ] ; NEW_LINE DEDENT DEDENT
for i in range ( n ) : NEW_LINE INDENT for j in range ( n ) : NEW_LINE INDENT print ' % 2d ▁ ' % ( arr [ i ] [ j ] ) , NEW_LINE DEDENT print NEW_LINE DEDENT
n = 8 NEW_LINE
DoublyEven ( n ) NEW_LINE
cola = 2 NEW_LINE rowa = 3 NEW_LINE colb = 3 NEW_LINE rowb = 2 NEW_LINE
def Kroneckerproduct ( A , B ) : NEW_LINE INDENT C = [ [ 0 for j in range ( cola * colb ) ] for i in range ( rowa * rowb ) ] NEW_LINE DEDENT
for i in range ( 0 , rowa ) : NEW_LINE
for k in range ( 0 , rowb ) : NEW_LINE
for j in range ( 0 , cola ) : NEW_LINE
for l in range ( 0 , colb ) : NEW_LINE
C [ i + l + 1 ] [ j + k + 1 ] = A [ i ] [ j ] * B [ k ] [ l ] NEW_LINE print ( C [ i + l + 1 ] [ j + k + 1 ] , end = ' ▁ ' ) NEW_LINE print ( " " ) NEW_LINE
A = [ [ 0 for j in range ( 2 ) ] for i in range ( 3 ) ] NEW_LINE B = [ [ 0 for j in range ( 3 ) ] for i in range ( 2 ) ] NEW_LINE A [ 0 ] [ 0 ] = 1 NEW_LINE A [ 0 ] [ 1 ] = 2 NEW_LINE A [ 1 ] [ 0 ] = 3 NEW_LINE A [ 1 ] [ 1 ] = 4 NEW_LINE A [ 2 ] [ 0 ] = 1 NEW_LINE A [ 2 ] [ 1 ] = 0 NEW_LINE B [ 0 ] [ 0 ] = 0 NEW_LINE B [ 0 ] [ 1 ] = 5 NEW_LINE B [ 0 ] [ 2 ] = 2 NEW_LINE B [ 1 ] [ 0 ] = 6 NEW_LINE B [ 1 ] [ 1 ] = 7 NEW_LINE B [ 1 ] [ 2 ] = 3 NEW_LINE Kroneckerproduct ( A , B ) NEW_LINE
def islowertriangular ( M ) : NEW_LINE INDENT for i in range ( 0 , len ( M ) ) : NEW_LINE INDENT for j in range ( i + 1 , len ( M ) ) : NEW_LINE INDENT if ( M [ i ] [ j ] != 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT DEDENT return True NEW_LINE DEDENT
M = [ [ 1 , 0 , 0 , 0 ] , [ 1 , 4 , 0 , 0 ] , [ 4 , 6 , 2 , 0 ] , [ 0 , 4 , 7 , 6 ] ] NEW_LINE
if islowertriangular ( M ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
def isuppertriangular ( M ) : NEW_LINE INDENT for i in range ( 1 , len ( M ) ) : NEW_LINE INDENT for j in range ( 0 , i ) : NEW_LINE INDENT if ( M [ i ] [ j ] != 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT DEDENT return True NEW_LINE DEDENT
M = [ [ 1 , 3 , 5 , 3 ] , [ 0 , 4 , 6 , 2 ] , [ 0 , 0 , 2 , 5 ] , [ 0 , 0 , 0 , 6 ] ] NEW_LINE if isuppertriangular ( M ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
m = 3 NEW_LINE
n = 2 NEW_LINE
def countSets ( a ) : NEW_LINE
res = 0 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT u = 0 NEW_LINE v = 0 NEW_LINE for j in range ( m ) : NEW_LINE INDENT if a [ i ] [ j ] : NEW_LINE INDENT u += 1 NEW_LINE DEDENT else : NEW_LINE INDENT v += 1 NEW_LINE DEDENT DEDENT res += pow ( 2 , u ) - 1 + pow ( 2 , v ) - 1 NEW_LINE DEDENT
for i in range ( m ) : NEW_LINE INDENT u = 0 NEW_LINE v = 0 NEW_LINE for j in range ( n ) : NEW_LINE INDENT if a [ j ] [ i ] : NEW_LINE INDENT u += 1 NEW_LINE DEDENT else : NEW_LINE INDENT v += 1 NEW_LINE DEDENT DEDENT res += pow ( 2 , u ) - 1 + pow ( 2 , v ) - 1 NEW_LINE DEDENT
return res - ( n * m ) NEW_LINE
a = [ [ 1 , 0 , 1 ] , [ 0 , 1 , 0 ] ] NEW_LINE print ( countSets ( a ) ) NEW_LINE
def transpose ( mat , tr , N ) : NEW_LINE INDENT for i in range ( N ) : NEW_LINE INDENT for j in range ( N ) : NEW_LINE INDENT tr [ i ] [ j ] = mat [ j ] [ i ] NEW_LINE DEDENT DEDENT DEDENT
def isSymmetric ( mat , N ) : NEW_LINE INDENT tr = [ [ 0 for j in range ( len ( mat [ 0 ] ) ) ] for i in range ( len ( mat ) ) ] NEW_LINE transpose ( mat , tr , N ) NEW_LINE for i in range ( N ) : NEW_LINE INDENT for j in range ( N ) : NEW_LINE INDENT if ( mat [ i ] [ j ] != tr [ i ] [ j ] ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT DEDENT return True NEW_LINE DEDENT
mat = [ [ 1 , 3 , 5 ] , [ 3 , 2 , 4 ] , [ 5 , 4 , 1 ] ] NEW_LINE if ( isSymmetric ( mat , 3 ) ) : NEW_LINE INDENT print " Yes " NEW_LINE DEDENT else : NEW_LINE INDENT print " No " NEW_LINE DEDENT
def isSymmetric ( mat , N ) : NEW_LINE INDENT for i in range ( N ) : NEW_LINE INDENT for j in range ( N ) : NEW_LINE INDENT if ( mat [ i ] [ j ] != mat [ j ] [ i ] ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT DEDENT return True NEW_LINE DEDENT
mat = [ [ 1 , 3 , 5 ] , [ 3 , 2 , 4 ] , [ 5 , 4 , 1 ] ] NEW_LINE if ( isSymmetric ( mat , 3 ) ) : NEW_LINE INDENT print " Yes " NEW_LINE DEDENT else : NEW_LINE INDENT print " No " NEW_LINE DEDENT
import math NEW_LINE
MAX = 100 ; NEW_LINE
def findNormal ( mat , n ) : NEW_LINE INDENT sum = 0 ; NEW_LINE for i in range ( n ) : NEW_LINE INDENT for j in range ( n ) : NEW_LINE INDENT sum += mat [ i ] [ j ] * mat [ i ] [ j ] ; NEW_LINE DEDENT DEDENT return math . floor ( math . sqrt ( sum ) ) ; NEW_LINE DEDENT
def findTrace ( mat , n ) : NEW_LINE INDENT sum = 0 ; NEW_LINE for i in range ( n ) : NEW_LINE INDENT sum += mat [ i ] [ i ] ; NEW_LINE DEDENT return sum ; NEW_LINE DEDENT
mat = [ [ 1 , 1 , 1 , 1 , 1 ] , [ 2 , 2 , 2 , 2 , 2 ] , [ 3 , 3 , 3 , 3 , 3 ] , [ 4 , 4 , 4 , 4 , 4 ] , [ 5 , 5 , 5 , 5 , 5 ] ] ; NEW_LINE print ( " Trace ▁ of ▁ Matrix ▁ = " , findTrace ( mat , 5 ) ) ; NEW_LINE print ( " Normal ▁ of ▁ Matrix ▁ = " , findNormal ( mat , 5 ) ) ; NEW_LINE
def maxDet ( n ) : NEW_LINE INDENT return 2 * n * n * n NEW_LINE DEDENT
def resMatrix ( n ) : NEW_LINE INDENT for i in range ( 3 ) : NEW_LINE INDENT for j in range ( 3 ) : NEW_LINE DEDENT DEDENT
if i == 0 and j == 2 : NEW_LINE INDENT print ( "0" , end = " ▁ " ) NEW_LINE DEDENT elif i == 1 and j == 0 : NEW_LINE INDENT print ( "0" , end = " ▁ " ) NEW_LINE DEDENT elif i == 2 and j == 1 : NEW_LINE INDENT print ( "0" , end = " ▁ " ) NEW_LINE DEDENT
else : NEW_LINE INDENT print ( n , end = " ▁ " ) NEW_LINE DEDENT print ( " " ) NEW_LINE
n = 15 NEW_LINE print ( " Maximum ▁ Detrminat = " , maxDet ( n ) ) NEW_LINE print ( " Resultant ▁ Matrix : " ) NEW_LINE resMatrix ( n ) NEW_LINE
def countNegative ( M , n , m ) : NEW_LINE INDENT count = 0 NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT for j in range ( m ) : NEW_LINE INDENT if M [ i ] [ j ] < 0 : NEW_LINE INDENT count += 1 NEW_LINE DEDENT else : NEW_LINE DEDENT DEDENT
break NEW_LINE return count NEW_LINE
M = [ [ - 3 , - 2 , - 1 , 1 ] , [ - 2 , 2 , 3 , 4 ] , [ 4 , 5 , 7 , 8 ] ] NEW_LINE print ( countNegative ( M , 3 , 4 ) ) NEW_LINE
def countNegative ( M , n , m ) : NEW_LINE
count = 0 NEW_LINE
i = 0 NEW_LINE j = m - 1 NEW_LINE
while j >= 0 and i < n : NEW_LINE INDENT if M [ i ] [ j ] < 0 : NEW_LINE DEDENT
count += ( j + 1 ) NEW_LINE
i += 1 NEW_LINE else : NEW_LINE
j -= 1 NEW_LINE return count NEW_LINE
M = [ [ - 3 , - 2 , - 1 , 1 ] , [ - 2 , 2 , 3 , 4 ] , [ 4 , 5 , 7 , 8 ] ] NEW_LINE print ( countNegative ( M , 3 , 4 ) ) NEW_LINE
def getLastNegativeIndex ( array , start , end , n ) : NEW_LINE
if ( start == end ) : NEW_LINE INDENT return start NEW_LINE DEDENT
mid = start + ( end - start ) // 2 NEW_LINE
if ( array [ mid ] < 0 ) : NEW_LINE
if ( mid + 1 < n and array [ mid + 1 ] >= 0 ) : NEW_LINE INDENT return mid NEW_LINE DEDENT
return getLastNegativeIndex ( array , mid + 1 , end , n ) NEW_LINE else : NEW_LINE
return getLastNegativeIndex ( array , start , mid - 1 , n ) NEW_LINE
def countNegative ( M , n , m ) : NEW_LINE
count = 0 NEW_LINE
nextEnd = m - 1 NEW_LINE
for i in range ( n ) : NEW_LINE
if ( M [ i ] [ 0 ] >= 0 ) : NEW_LINE INDENT break NEW_LINE DEDENT
nextEnd = getLastNegativeIndex ( M [ i ] , 0 , nextEnd , 4 ) NEW_LINE count += nextEnd + 1 NEW_LINE return count NEW_LINE
M = [ [ - 3 , - 2 , - 1 , 1 ] , [ - 2 , 2 , 3 , 4 ] , [ 4 , 5 , 7 , 8 ] ] NEW_LINE r = 3 NEW_LINE c = 4 NEW_LINE print ( countNegative ( M , r , c ) ) NEW_LINE
N = 5 NEW_LINE
def findMaxValue ( mat ) : NEW_LINE
maxValue = 0 NEW_LINE
for a in range ( N - 1 ) : NEW_LINE INDENT for b in range ( N - 1 ) : NEW_LINE INDENT for d in range ( a + 1 , N ) : NEW_LINE INDENT for e in range ( b + 1 , N ) : NEW_LINE INDENT if maxValue < int ( mat [ d ] [ e ] - mat [ a ] [ b ] ) : NEW_LINE INDENT maxValue = int ( mat [ d ] [ e ] - mat [ a ] [ b ] ) ; NEW_LINE DEDENT DEDENT DEDENT DEDENT DEDENT return maxValue ; NEW_LINE
mat = [ [ 1 , 2 , - 1 , - 4 , - 20 ] , [ - 8 , - 3 , 4 , 2 , 1 ] , [ 3 , 8 , 6 , 1 , 3 ] , [ - 4 , - 1 , 1 , 7 , - 6 ] , [ 0 , - 4 , 10 , - 5 , 1 ] ] ; NEW_LINE print ( " Maximum ▁ Value ▁ is ▁ " + str ( findMaxValue ( mat ) ) ) NEW_LINE
import sys NEW_LINE N = 5 NEW_LINE
def findMaxValue ( mat ) : NEW_LINE
maxValue = - sys . maxsize - 1 NEW_LINE
maxArr = [ [ 0 for x in range ( N ) ] for y in range ( N ) ] NEW_LINE
maxArr [ N - 1 ] [ N - 1 ] = mat [ N - 1 ] [ N - 1 ] NEW_LINE
maxv = mat [ N - 1 ] [ N - 1 ] ; NEW_LINE for j in range ( N - 2 , - 1 , - 1 ) : NEW_LINE INDENT if ( mat [ N - 1 ] [ j ] > maxv ) : NEW_LINE INDENT maxv = mat [ N - 1 ] [ j ] NEW_LINE DEDENT maxArr [ N - 1 ] [ j ] = maxv NEW_LINE DEDENT
maxv = mat [ N - 1 ] [ N - 1 ] ; NEW_LINE for i in range ( N - 2 , - 1 , - 1 ) : NEW_LINE INDENT if ( mat [ i ] [ N - 1 ] > maxv ) : NEW_LINE INDENT maxv = mat [ i ] [ N - 1 ] NEW_LINE DEDENT maxArr [ i ] [ N - 1 ] = maxv NEW_LINE DEDENT
for i in range ( N - 2 , - 1 , - 1 ) : NEW_LINE INDENT for j in range ( N - 2 , - 1 , - 1 ) : NEW_LINE DEDENT
if ( maxArr [ i + 1 ] [ j + 1 ] - mat [ i ] [ j ] > maxValue ) : NEW_LINE INDENT maxValue = ( maxArr [ i + 1 ] [ j + 1 ] - mat [ i ] [ j ] ) NEW_LINE DEDENT
maxArr [ i ] [ j ] = max ( mat [ i ] [ j ] , max ( maxArr [ i ] [ j + 1 ] , maxArr [ i + 1 ] [ j ] ) ) NEW_LINE return maxValue NEW_LINE
mat = [ [ 1 , 2 , - 1 , - 4 , - 20 ] , [ - 8 , - 3 , 4 , 2 , 1 ] , [ 3 , 8 , 6 , 1 , 3 ] , [ - 4 , - 1 , 1 , 7 , - 6 ] , [ 0 , - 4 , 10 , - 5 , 1 ] ] NEW_LINE print ( " Maximum ▁ Value ▁ is " , findMaxValue ( mat ) ) NEW_LINE
import sys NEW_LINE INF = sys . maxsize NEW_LINE N = 4 NEW_LINE
def youngify ( mat , i , j ) : NEW_LINE
downVal = mat [ i + 1 ] [ j ] if ( i + 1 < N ) else INF NEW_LINE rightVal = mat [ i ] [ j + 1 ] if ( j + 1 < N ) else INF NEW_LINE
if ( downVal == INF and rightVal == INF ) : NEW_LINE INDENT return NEW_LINE DEDENT
if ( downVal < rightVal ) : NEW_LINE INDENT mat [ i ] [ j ] = downVal NEW_LINE mat [ i + 1 ] [ j ] = INF NEW_LINE youngify ( mat , i + 1 , j ) NEW_LINE DEDENT else : NEW_LINE INDENT mat [ i ] [ j ] = rightVal NEW_LINE mat [ i ] [ j + 1 ] = INF NEW_LINE youngify ( mat , i , j + 1 ) NEW_LINE DEDENT
def extractMin ( mat ) : NEW_LINE INDENT ret = mat [ 0 ] [ 0 ] NEW_LINE mat [ 0 ] [ 0 ] = INF NEW_LINE youngify ( mat , 0 , 0 ) NEW_LINE return ret NEW_LINE DEDENT
def printSorted ( mat ) : NEW_LINE INDENT print ( " Elements ▁ of ▁ matrix ▁ in ▁ sorted ▁ order ▁ n " ) NEW_LINE i = 0 NEW_LINE while i < N * N : NEW_LINE INDENT print ( extractMin ( mat ) , end = " ▁ " ) NEW_LINE i += 1 NEW_LINE DEDENT DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT mat = [ [ 10 , 20 , 30 , 40 ] , [ 15 , 25 , 35 , 45 ] , [ 27 , 29 , 37 , 48 ] , [ 32 , 33 , 39 , 50 ] ] NEW_LINE printSorted ( mat ) NEW_LINE DEDENT
n = 5 NEW_LINE
def printSumSimple ( mat , k ) : NEW_LINE
if ( k > n ) : NEW_LINE INDENT return NEW_LINE DEDENT
for i in range ( n - k + 1 ) : NEW_LINE
for j in range ( n - k + 1 ) : NEW_LINE
sum = 0 NEW_LINE for p in range ( i , k + i ) : NEW_LINE INDENT for q in range ( j , k + j ) : NEW_LINE INDENT sum += mat [ p ] [ q ] NEW_LINE DEDENT DEDENT print ( sum , end = " ▁ " ) NEW_LINE
print ( ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT mat = [ [ 1 , 1 , 1 , 1 , 1 ] , [ 2 , 2 , 2 , 2 , 2 ] , [ 3 , 3 , 3 , 3 , 3 ] , [ 4 , 4 , 4 , 4 , 4 ] , [ 5 , 5 , 5 , 5 , 5 ] ] NEW_LINE k = 3 NEW_LINE printSumSimple ( mat , k ) NEW_LINE DEDENT
n = 5 NEW_LINE
def printSumTricky ( mat , k ) : NEW_LINE INDENT global n NEW_LINE DEDENT
if k > n : NEW_LINE INDENT return NEW_LINE DEDENT
stripSum = [ [ None ] * n for i in range ( n ) ] NEW_LINE
for j in range ( n ) : NEW_LINE
Sum = 0 NEW_LINE for i in range ( k ) : NEW_LINE INDENT Sum += mat [ i ] [ j ] NEW_LINE DEDENT stripSum [ 0 ] [ j ] = Sum NEW_LINE
for i in range ( 1 , n - k + 1 ) : NEW_LINE INDENT Sum += ( mat [ i + k - 1 ] [ j ] - mat [ i - 1 ] [ j ] ) NEW_LINE stripSum [ i ] [ j ] = Sum NEW_LINE DEDENT
for i in range ( n - k + 1 ) : NEW_LINE
Sum = 0 NEW_LINE for j in range ( k ) : NEW_LINE INDENT Sum += stripSum [ i ] [ j ] NEW_LINE DEDENT print ( Sum , end = " ▁ " ) NEW_LINE
for j in range ( 1 , n - k + 1 ) : NEW_LINE INDENT Sum += ( stripSum [ i ] [ j + k - 1 ] - stripSum [ i ] [ j - 1 ] ) NEW_LINE print ( Sum , end = " ▁ " ) NEW_LINE DEDENT print ( ) NEW_LINE
mat = [ [ 1 , 1 , 1 , 1 , 1 ] , [ 2 , 2 , 2 , 2 , 2 ] , [ 3 , 3 , 3 , 3 , 3 ] , [ 4 , 4 , 4 , 4 , 4 ] , [ 5 , 5 , 5 , 5 , 5 ] ] NEW_LINE k = 3 NEW_LINE printSumTricky ( mat , k ) NEW_LINE
M = 3 NEW_LINE N = 4 NEW_LINE
def transpose ( A , B ) : NEW_LINE INDENT for i in range ( N ) : NEW_LINE INDENT for j in range ( M ) : NEW_LINE INDENT B [ i ] [ j ] = A [ j ] [ i ] NEW_LINE DEDENT DEDENT DEDENT
A = [ [ 1 , 1 , 1 , 1 ] , [ 2 , 2 , 2 , 2 ] , [ 3 , 3 , 3 , 3 ] ] NEW_LINE B = [ [ 0 for x in range ( M ) ] for y in range ( N ) ] NEW_LINE transpose ( A , B ) NEW_LINE print ( " Result ▁ matrix ▁ is " ) NEW_LINE for i in range ( N ) : NEW_LINE INDENT for j in range ( M ) : NEW_LINE INDENT print ( B [ i ] [ j ] , " ▁ " , end = ' ' ) NEW_LINE DEDENT print ( ) NEW_LINE DEDENT
N = 4 NEW_LINE
def transpose ( A ) : NEW_LINE INDENT for i in range ( N ) : NEW_LINE INDENT for j in range ( i + 1 , N ) : NEW_LINE INDENT A [ i ] [ j ] , A [ j ] [ i ] = A [ j ] [ i ] , A [ i ] [ j ] NEW_LINE DEDENT DEDENT DEDENT
A = [ [ 1 , 1 , 1 , 1 ] , [ 2 , 2 , 2 , 2 ] , [ 3 , 3 , 3 , 3 ] , [ 4 , 4 , 4 , 4 ] ] NEW_LINE transpose ( A ) NEW_LINE print ( " Modified ▁ matrix ▁ is " ) NEW_LINE for i in range ( N ) : NEW_LINE INDENT for j in range ( N ) : NEW_LINE INDENT print ( A [ i ] [ j ] , " ▁ " , end = ' ' ) NEW_LINE DEDENT print ( ) NEW_LINE DEDENT
R = 3 NEW_LINE C = 3 NEW_LINE
def pathCountRec ( mat , m , n , k ) : NEW_LINE
if m < 0 or n < 0 : NEW_LINE INDENT return 0 NEW_LINE DEDENT elif m == 0 and n == 0 : NEW_LINE INDENT return k == mat [ m ] [ n ] NEW_LINE DEDENT
return ( pathCountRec ( mat , m - 1 , n , k - mat [ m ] [ n ] ) + pathCountRec ( mat , m , n - 1 , k - mat [ m ] [ n ] ) ) NEW_LINE
def pathCount ( mat , k ) : NEW_LINE INDENT return pathCountRec ( mat , R - 1 , C - 1 , k ) NEW_LINE DEDENT
k = 12 NEW_LINE mat = [ [ 1 , 2 , 3 ] , [ 4 , 6 , 5 ] , [ 3 , 2 , 1 ] ] NEW_LINE print ( pathCount ( mat , k ) ) NEW_LINE
R = 3 NEW_LINE C = 3 NEW_LINE MAX_K = 1000 NEW_LINE def pathCountDPRecDP ( mat , m , n , k ) : NEW_LINE
if m < 0 or n < 0 : NEW_LINE INDENT return 0 NEW_LINE DEDENT elif m == 0 and n == 0 : NEW_LINE INDENT return k == mat [ m ] [ n ] NEW_LINE DEDENT
if ( dp [ m ] [ n ] [ k ] != - 1 ) : NEW_LINE INDENT return dp [ m ] [ n ] [ k ] NEW_LINE DEDENT
dp [ m ] [ n ] [ k ] = ( pathCountDPRecDP ( mat , m - 1 , n , k - mat [ m ] [ n ] ) + pathCountDPRecDP ( mat , m , n - 1 , k - mat [ m ] [ n ] ) ) NEW_LINE return dp [ m ] [ n ] [ k ] NEW_LINE
def pathCountDP ( mat , k ) : NEW_LINE INDENT return pathCountDPRecDP ( mat , R - 1 , C - 1 , k ) NEW_LINE DEDENT
k = 12 NEW_LINE dp = [ [ [ - 1 for col in range ( MAX_K ) ] for col in range ( C ) ] for row in range ( R ) ] NEW_LINE mat = [ [ 1 , 2 , 3 ] , [ 4 , 6 , 5 ] , [ 3 , 2 , 1 ] ] NEW_LINE print ( pathCountDP ( mat , k ) ) NEW_LINE
SIZE = 10 NEW_LINE
def sortMat ( mat , n ) : NEW_LINE
temp = [ 0 ] * ( n * n ) NEW_LINE k = 0 NEW_LINE
for i in range ( 0 , n ) : NEW_LINE INDENT for j in range ( 0 , n ) : NEW_LINE INDENT temp [ k ] = mat [ i ] [ j ] NEW_LINE k += 1 NEW_LINE DEDENT DEDENT
temp . sort ( ) NEW_LINE
k = 0 NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT for j in range ( 0 , n ) : NEW_LINE INDENT mat [ i ] [ j ] = temp [ k ] NEW_LINE k += 1 NEW_LINE DEDENT DEDENT
def printMat ( mat , n ) : NEW_LINE INDENT for i in range ( 0 , n ) : NEW_LINE INDENT for j in range ( 0 , n ) : NEW_LINE INDENT print ( mat [ i ] [ j ] , end = " ▁ " ) NEW_LINE DEDENT print ( ) NEW_LINE DEDENT DEDENT
mat = [ [ 5 , 4 , 7 ] , [ 1 , 3 , 8 ] , [ 2 , 9 , 6 ] ] NEW_LINE n = 3 NEW_LINE print ( " Original ▁ Matrix : " ) NEW_LINE printMat ( mat , n ) NEW_LINE sortMat ( mat , n ) NEW_LINE print ( " Matrix After Sorting : " ) NEW_LINE printMat ( mat , n ) NEW_LINE
def bubbleSort ( arr ) : NEW_LINE INDENT n = len ( arr ) NEW_LINE for i in range ( n ) : NEW_LINE INDENT swapped = False NEW_LINE for j in range ( 0 , n - i - 1 ) : NEW_LINE INDENT if arr [ j ] > arr [ j + 1 ] : NEW_LINE DEDENT DEDENT DEDENT
arr [ j ] , arr [ j + 1 ] = arr [ j + 1 ] , arr [ j ] NEW_LINE swapped = True NEW_LINE
if swapped == False : NEW_LINE INDENT break NEW_LINE DEDENT
arr = [ 64 , 34 , 25 , 12 , 22 , 11 , 90 ] NEW_LINE bubbleSort ( arr ) NEW_LINE print ( " Sorted ▁ array ▁ : " ) NEW_LINE for i in range ( len ( arr ) ) : NEW_LINE INDENT print ( " % d " % arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT
def findCrossOver ( arr , low , high , x ) : NEW_LINE
if ( arr [ high ] <= x ) : NEW_LINE INDENT return high NEW_LINE DEDENT
if ( arr [ low ] > x ) : NEW_LINE INDENT return low NEW_LINE DEDENT
mid = ( low + high ) // 2 NEW_LINE
if ( arr [ mid ] <= x and arr [ mid + 1 ] > x ) : NEW_LINE INDENT return mid NEW_LINE DEDENT
if ( arr [ mid ] < x ) : NEW_LINE INDENT return findCrossOver ( arr , mid + 1 , high , x ) NEW_LINE DEDENT return findCrossOver ( arr , low , mid - 1 , x ) NEW_LINE
def printKclosest ( arr , x , k , n ) : NEW_LINE
l = findCrossOver ( arr , 0 , n - 1 , x ) NEW_LINE
r = l + 1 NEW_LINE
count = 0 NEW_LINE
if ( arr [ l ] == x ) : NEW_LINE INDENT l -= 1 NEW_LINE DEDENT
while ( l >= 0 and r < n and count < k ) : NEW_LINE INDENT if ( x - arr [ l ] < arr [ r ] - x ) : NEW_LINE INDENT print ( arr [ l ] , end = " ▁ " ) NEW_LINE l -= 1 NEW_LINE DEDENT else : NEW_LINE INDENT print ( arr [ r ] , end = " ▁ " ) NEW_LINE r += 1 NEW_LINE DEDENT count += 1 NEW_LINE DEDENT
while ( count < k and l >= 0 ) : NEW_LINE INDENT print ( arr [ l ] , end = " ▁ " ) NEW_LINE l -= 1 NEW_LINE count += 1 NEW_LINE DEDENT
while ( count < k and r < n ) : NEW_LINE INDENT print ( arr [ r ] , end = " ▁ " ) NEW_LINE r += 1 NEW_LINE count += 1 NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 12 , 16 , 22 , 30 , 35 , 39 , 42 , 45 , 48 , 50 , 53 , 55 , 56 ] NEW_LINE n = len ( arr ) NEW_LINE x = 35 NEW_LINE k = 4 NEW_LINE printKclosest ( arr , x , 4 , n ) NEW_LINE DEDENT
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . next = None NEW_LINE DEDENT DEDENT
def push ( head_ref , new_data ) : NEW_LINE
new_node = Node ( 0 ) NEW_LINE new_node . data = new_data NEW_LINE
new_node . next = ( head_ref ) NEW_LINE
( head_ref ) = new_node NEW_LINE return head_ref NEW_LINE
def insertionSort ( head_ref ) : NEW_LINE
sorted = None NEW_LINE
current = head_ref NEW_LINE while ( current != None ) : NEW_LINE
next = current . next NEW_LINE
sorted = sortedInsert ( sorted , current ) NEW_LINE
current = next NEW_LINE
head_ref = sorted NEW_LINE return head_ref NEW_LINE
def sortedInsert ( head_ref , new_node ) : NEW_LINE INDENT current = None NEW_LINE DEDENT
if ( head_ref == None or ( head_ref ) . data >= new_node . data ) : NEW_LINE INDENT new_node . next = head_ref NEW_LINE head_ref = new_node NEW_LINE DEDENT else : NEW_LINE INDENT current = head_ref NEW_LINE DEDENT
while ( current . next != None and current . next . data < new_node . data ) : NEW_LINE INDENT current = current . next NEW_LINE DEDENT new_node . next = current . next NEW_LINE current . next = new_node NEW_LINE return head_ref NEW_LINE
def printList ( head ) : NEW_LINE INDENT temp = head NEW_LINE while ( temp != None ) : NEW_LINE INDENT print ( temp . data , end = " ▁ " ) NEW_LINE temp = temp . next NEW_LINE DEDENT DEDENT
a = None NEW_LINE a = push ( a , 5 ) NEW_LINE a = push ( a , 20 ) NEW_LINE a = push ( a , 4 ) NEW_LINE a = push ( a , 3 ) NEW_LINE a = push ( a , 30 ) NEW_LINE print ( " Linked ▁ List ▁ before ▁ sorting ▁ " ) NEW_LINE printList ( a ) NEW_LINE a = insertionSort ( a ) NEW_LINE print ( " Linked List after sorting   " ) NEW_LINE printList ( a ) NEW_LINE
def count ( S , m , n ) : NEW_LINE
table = [ 0 for k in range ( n + 1 ) ] NEW_LINE
table [ 0 ] = 1 NEW_LINE
for i in range ( 0 , m ) : NEW_LINE INDENT for j in range ( S [ i ] , n + 1 ) : NEW_LINE INDENT table [ j ] += table [ j - S [ i ] ] NEW_LINE DEDENT DEDENT return table [ n ] NEW_LINE
import sys NEW_LINE dp = [ [ - 1 for i in range ( 100 ) ] for j in range ( 100 ) ] NEW_LINE
def matrixChainMemoised ( p , i , j ) : NEW_LINE INDENT if ( i == j ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT if ( dp [ i ] [ j ] != - 1 ) : NEW_LINE INDENT return dp [ i ] [ j ] NEW_LINE DEDENT dp [ i ] [ j ] = sys . maxsize NEW_LINE for k in range ( i , j ) : NEW_LINE INDENT dp [ i ] [ j ] = min ( dp [ i ] [ j ] , matrixChainMemoised ( p , i , k ) + matrixChainMemoised ( p , k + 1 , j ) + p [ i - 1 ] * p [ k ] * p [ j ] ) NEW_LINE DEDENT return dp [ i ] [ j ] NEW_LINE DEDENT def MatrixChainOrder ( p , n ) : NEW_LINE INDENT i = 1 NEW_LINE j = n - 1 NEW_LINE return matrixChainMemoised ( p , i , j ) NEW_LINE DEDENT
arr = [ 1 , 2 , 3 , 4 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Minimum ▁ number ▁ of ▁ multiplications ▁ is " , MatrixChainOrder ( arr , n ) ) NEW_LINE
import sys NEW_LINE
def MatrixChainOrder ( p , n ) : NEW_LINE
m = [ [ 0 for x in range ( n ) ] for x in range ( n ) ] NEW_LINE
for i in range ( 1 , n ) : NEW_LINE INDENT m [ i ] [ i ] = 0 NEW_LINE DEDENT
for L in range ( 2 , n ) : NEW_LINE INDENT for i in range ( 1 , n - L + 1 ) : NEW_LINE INDENT j = i + L - 1 NEW_LINE m [ i ] [ j ] = sys . maxint NEW_LINE for k in range ( i , j ) : NEW_LINE DEDENT DEDENT
q = m [ i ] [ k ] + m [ k + 1 ] [ j ] + p [ i - 1 ] * p [ k ] * p [ j ] NEW_LINE if q < m [ i ] [ j ] : NEW_LINE INDENT m [ i ] [ j ] = q NEW_LINE DEDENT return m [ 1 ] [ n - 1 ] NEW_LINE
arr = [ 1 , 2 , 3 , 4 ] NEW_LINE size = len ( arr ) NEW_LINE print ( " Minimum ▁ number ▁ of ▁ multiplications ▁ is ▁ " + str ( MatrixChainOrder ( arr , size ) ) ) NEW_LINE
import sys NEW_LINE
def cutRod ( price , n ) : NEW_LINE INDENT if ( n <= 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT max_val = - sys . maxsize - 1 NEW_LINE DEDENT
for i in range ( 0 , n ) : NEW_LINE INDENT max_val = max ( max_val , price [ i ] + cutRod ( price , n - i - 1 ) ) NEW_LINE DEDENT return max_val NEW_LINE
arr = [ 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 ] NEW_LINE size = len ( arr ) NEW_LINE print ( " Maximum ▁ Obtainable ▁ Value ▁ is " , cutRod ( arr , size ) ) NEW_LINE
INT_MIN = - 32767 NEW_LINE
def cutRod ( price , n ) : NEW_LINE INDENT val = [ 0 for x in range ( n + 1 ) ] NEW_LINE val [ 0 ] = 0 NEW_LINE DEDENT
for i in range ( 1 , n + 1 ) : NEW_LINE INDENT max_val = INT_MIN NEW_LINE for j in range ( i ) : NEW_LINE INDENT max_val = max ( max_val , price [ j ] + val [ i - j - 1 ] ) NEW_LINE DEDENT val [ i ] = max_val NEW_LINE DEDENT return val [ n ] NEW_LINE
arr = [ 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 ] NEW_LINE size = len ( arr ) NEW_LINE print ( " Maximum ▁ Obtainable ▁ Value ▁ is ▁ " + str ( cutRod ( arr , size ) ) ) NEW_LINE
def multiply ( x , y ) : NEW_LINE
if ( y == 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
if ( y > 0 ) : NEW_LINE INDENT return ( x + multiply ( x , y - 1 ) ) NEW_LINE DEDENT
if ( y < 0 ) : NEW_LINE INDENT return - multiply ( x , - y ) NEW_LINE DEDENT
print ( multiply ( 5 , - 11 ) ) NEW_LINE
def SieveOfEratosthenes ( n ) : NEW_LINE
prime = [ True for i in range ( n + 1 ) ] NEW_LINE p = 2 NEW_LINE while ( p * p <= n ) : NEW_LINE
if ( prime [ p ] == True ) : NEW_LINE
for i in range ( p * p , n + 1 , p ) : NEW_LINE INDENT prime [ i ] = False NEW_LINE DEDENT p += 1 NEW_LINE
for p in range ( 2 , n + 1 ) : NEW_LINE INDENT if prime [ p ] : NEW_LINE INDENT print p , NEW_LINE DEDENT DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 30 NEW_LINE print " Following ▁ are ▁ the ▁ prime ▁ numbers ▁ smaller " , NEW_LINE print " than ▁ or ▁ equal ▁ to " , n NEW_LINE SieveOfEratosthenes ( n ) NEW_LINE DEDENT
def binomialCoeff ( n , k ) : NEW_LINE INDENT res = 1 NEW_LINE if ( k > n - k ) : NEW_LINE INDENT k = n - k NEW_LINE DEDENT for i in range ( 0 , k ) : NEW_LINE INDENT res = res * ( n - i ) NEW_LINE res = res // ( i + 1 ) NEW_LINE DEDENT return res NEW_LINE DEDENT
def printPascal ( n ) : NEW_LINE
for line in range ( 0 , n ) : NEW_LINE
for i in range ( 0 , line + 1 ) : NEW_LINE INDENT print ( binomialCoeff ( line , i ) , " ▁ " , end = " " ) NEW_LINE DEDENT print ( ) NEW_LINE
n = 7 NEW_LINE printPascal ( n ) NEW_LINE
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
def getModulo ( n , d ) : NEW_LINE INDENT return ( n & ( d - 1 ) ) NEW_LINE DEDENT
n = 6 NEW_LINE
d = 4 NEW_LINE print ( n , " moduo " , d , " is " , getModulo ( n , d ) ) NEW_LINE
def countSetBits ( n ) : NEW_LINE INDENT count = 0 NEW_LINE while ( n ) : NEW_LINE INDENT count += n & 1 NEW_LINE n >>= 1 NEW_LINE DEDENT return count NEW_LINE DEDENT
i = 9 NEW_LINE print ( countSetBits ( i ) ) NEW_LINE
def countSetBits ( n ) : NEW_LINE
if ( n == 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT else : NEW_LINE INDENT return 1 + countSetBits ( n & ( n - 1 ) ) NEW_LINE DEDENT
n = 9 NEW_LINE
print ( countSetBits ( n ) ) NEW_LINE
BitsSetTable256 = [ 0 ] * 256 NEW_LINE
def initialize ( ) : NEW_LINE
BitsSetTable256 [ 0 ] = 0 NEW_LINE for i in range ( 256 ) : NEW_LINE INDENT BitsSetTable256 [ i ] = ( i & 1 ) + BitsSetTable256 [ i // 2 ] NEW_LINE DEDENT
def countSetBits ( n ) : NEW_LINE INDENT return ( BitsSetTable256 [ n & 0xff ] + BitsSetTable256 [ ( n >> 8 ) & 0xff ] + BitsSetTable256 [ ( n >> 16 ) & 0xff ] + BitsSetTable256 [ n >> 24 ] ) NEW_LINE DEDENT
initialize ( ) NEW_LINE n = 9 NEW_LINE print ( countSetBits ( n ) ) NEW_LINE
print ( bin ( 4 ) . count ( '1' ) ) ; NEW_LINE print ( bin ( 15 ) . count ( '1' ) ) ; NEW_LINE
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
def powerof2 ( n ) : NEW_LINE
if n == 1 : NEW_LINE INDENT return True NEW_LINE DEDENT
elif n % 2 != 0 or n == 0 : NEW_LINE INDENT return False NEW_LINE DEDENT
return powerof2 ( n / 2 ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
print ( powerof2 ( 64 ) ) NEW_LINE
print ( powerof2 ( 12 ) ) NEW_LINE
def isPowerOfTwo ( x ) : NEW_LINE
return ( x and ( not ( x & ( x - 1 ) ) ) ) NEW_LINE
if ( isPowerOfTwo ( 31 ) ) : NEW_LINE INDENT print ( ' Yes ' ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( ' No ' ) NEW_LINE DEDENT if ( isPowerOfTwo ( 64 ) ) : NEW_LINE INDENT print ( ' Yes ' ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( ' No ' ) NEW_LINE DEDENT
def maxRepeating ( arr , n , k ) : NEW_LINE
for i in range ( 0 , n ) : NEW_LINE INDENT arr [ arr [ i ] % k ] += k NEW_LINE DEDENT
max = arr [ 0 ] NEW_LINE result = 0 NEW_LINE for i in range ( 1 , n ) : NEW_LINE INDENT if arr [ i ] > max : NEW_LINE INDENT max = arr [ i ] NEW_LINE result = i NEW_LINE DEDENT DEDENT
return result NEW_LINE
arr = [ 2 , 3 , 3 , 5 , 3 , 4 , 1 , 7 ] NEW_LINE n = len ( arr ) NEW_LINE k = 8 NEW_LINE print ( " The ▁ maximum ▁ repeating ▁ number ▁ is " , maxRepeating ( arr , n , k ) ) NEW_LINE
def fun ( x ) : NEW_LINE INDENT y = ( x // 4 ) * 4 NEW_LINE DEDENT
ans = 0 NEW_LINE for i in range ( y , x + 1 ) : NEW_LINE INDENT ans ^= i NEW_LINE DEDENT return ans NEW_LINE
def query ( x ) : NEW_LINE
if ( x == 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT k = ( x + 1 ) // 2 NEW_LINE
if x % 2 == 0 : NEW_LINE INDENT return ( ( fun ( k - 1 ) * 2 ) ^ ( k & 1 ) ) NEW_LINE DEDENT else : NEW_LINE INDENT return ( 2 * fun ( k ) ) NEW_LINE DEDENT def allQueries ( q , l , r ) : NEW_LINE for i in range ( q ) : NEW_LINE INDENT print ( query ( r [ i ] ) ^ query ( l [ i ] - 1 ) ) NEW_LINE DEDENT
q = 3 NEW_LINE l = [ 2 , 2 , 5 ] NEW_LINE r = [ 4 , 8 , 9 ] NEW_LINE allQueries ( q , l , r ) NEW_LINE
def prefixXOR ( arr , preXOR , n ) : NEW_LINE
for i in range ( 0 , n , 1 ) : NEW_LINE INDENT while ( arr [ i ] % 2 != 1 ) : NEW_LINE INDENT arr [ i ] = int ( arr [ i ] / 2 ) NEW_LINE DEDENT preXOR [ i ] = arr [ i ] NEW_LINE DEDENT
for i in range ( 1 , n , 1 ) : NEW_LINE INDENT preXOR [ i ] = preXOR [ i - 1 ] ^ preXOR [ i ] NEW_LINE DEDENT
def query ( preXOR , l , r ) : NEW_LINE INDENT if ( l == 0 ) : NEW_LINE INDENT return preXOR [ r ] NEW_LINE DEDENT else : NEW_LINE INDENT return preXOR [ r ] ^ preXOR [ l - 1 ] NEW_LINE DEDENT DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 3 , 4 , 5 ] NEW_LINE n = len ( arr ) NEW_LINE preXOR = [ 0 for i in range ( n ) ] NEW_LINE prefixXOR ( arr , preXOR , n ) NEW_LINE print ( query ( preXOR , 0 , 2 ) ) NEW_LINE print ( query ( preXOR , 1 , 2 ) ) NEW_LINE DEDENT
def findMinSwaps ( arr , n ) : NEW_LINE
noOfZeroes = [ 0 ] * n NEW_LINE count = 0 NEW_LINE
noOfZeroes [ n - 1 ] = 1 - arr [ n - 1 ] NEW_LINE for i in range ( n - 2 , - 1 , - 1 ) : NEW_LINE INDENT noOfZeroes [ i ] = noOfZeroes [ i + 1 ] NEW_LINE if ( arr [ i ] == 0 ) : NEW_LINE INDENT noOfZeroes [ i ] = noOfZeroes [ i ] + 1 NEW_LINE DEDENT DEDENT
for i in range ( 0 , n ) : NEW_LINE INDENT if ( arr [ i ] == 1 ) : NEW_LINE INDENT count = count + noOfZeroes [ i ] NEW_LINE DEDENT DEDENT return count NEW_LINE
arr = [ 0 , 0 , 1 , 0 , 1 , 0 , 1 , 1 ] NEW_LINE n = len ( arr ) NEW_LINE print ( findMinSwaps ( arr , n ) ) NEW_LINE
def minswaps ( arr ) : NEW_LINE INDENT count = 0 NEW_LINE num_unplaced_zeros = 0 NEW_LINE for index in range ( len ( arr ) - 1 , - 1 , - 1 ) : NEW_LINE INDENT if arr [ index ] == 0 : NEW_LINE INDENT num_unplaced_zeros += 1 NEW_LINE DEDENT else : NEW_LINE INDENT count += num_unplaced_zeros NEW_LINE DEDENT DEDENT return count NEW_LINE DEDENT
arr = [ 0 , 0 , 1 , 0 , 1 , 0 , 1 , 1 ] NEW_LINE print ( minswaps ( arr ) ) NEW_LINE
def arraySortedOrNot ( arr , n ) : NEW_LINE
if ( n == 0 or n == 1 ) : NEW_LINE INDENT return True NEW_LINE DEDENT for i in range ( 1 , n ) : NEW_LINE
if ( arr [ i - 1 ] > arr [ i ] ) : NEW_LINE INDENT return False NEW_LINE DEDENT
return True NEW_LINE
arr = [ 20 , 23 , 23 , 45 , 78 , 88 ] NEW_LINE n = len ( arr ) NEW_LINE if ( arraySortedOrNot ( arr , n ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
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
def printMax ( arr , k , n ) : NEW_LINE
brr = arr . copy ( ) NEW_LINE
brr . sort ( reverse = True ) NEW_LINE
for i in range ( n ) : NEW_LINE INDENT if ( arr [ i ] in brr [ 0 : k ] ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT DEDENT
arr = [ 50 , 8 , 45 , 12 , 25 , 40 , 84 ] NEW_LINE n = len ( arr ) NEW_LINE k = 3 NEW_LINE printMax ( arr , k , n ) NEW_LINE
def printSmall ( arr , asize , n ) : NEW_LINE
copy_arr = arr . copy ( ) NEW_LINE
copy_arr . sort ( ) NEW_LINE
for i in range ( asize ) : NEW_LINE INDENT if binary_search ( copy_arr , low = 0 , high = n , ele = arr [ i ] ) > - 1 : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 1 , 5 , 8 , 9 , 6 , 7 , 3 , 4 , 2 , 0 ] NEW_LINE asize = len ( arr ) NEW_LINE n = 5 NEW_LINE printSmall ( arr , asize , n ) NEW_LINE DEDENT
def checkIsAP ( arr , n ) : NEW_LINE INDENT if ( n == 1 ) : return True NEW_LINE DEDENT
arr . sort ( ) NEW_LINE
d = arr [ 1 ] - arr [ 0 ] NEW_LINE for i in range ( 2 , n ) : NEW_LINE INDENT if ( arr [ i ] - arr [ i - 1 ] != d ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT return True NEW_LINE
arr = [ 20 , 15 , 5 , 0 , 10 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Yes " ) if ( checkIsAP ( arr , n ) ) else print ( " No " ) NEW_LINE
def countPairs ( a , n ) : NEW_LINE
mn = + 2147483647 NEW_LINE mx = - 2147483648 NEW_LINE for i in range ( n ) : NEW_LINE INDENT mn = min ( mn , a [ i ] ) NEW_LINE mx = max ( mx , a [ i ] ) NEW_LINE DEDENT
c1 = 0 NEW_LINE
c2 = 0 NEW_LINE for i in range ( n ) : NEW_LINE INDENT if ( a [ i ] == mn ) : NEW_LINE INDENT c1 += 1 NEW_LINE DEDENT if ( a [ i ] == mx ) : NEW_LINE INDENT c2 += 1 NEW_LINE DEDENT DEDENT
if ( mn == mx ) : NEW_LINE INDENT return n * ( n - 1 ) // 2 NEW_LINE DEDENT else : NEW_LINE INDENT return c1 * c2 NEW_LINE DEDENT
a = [ 3 , 2 , 1 , 1 , 3 ] NEW_LINE n = len ( a ) NEW_LINE print ( countPairs ( a , n ) ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , x ) : NEW_LINE INDENT self . data = x NEW_LINE self . next = None NEW_LINE DEDENT DEDENT
def rearrange ( head ) : NEW_LINE
if ( head == None ) : NEW_LINE INDENT return head NEW_LINE DEDENT
prev , curr = head , head . next NEW_LINE while ( curr ) : NEW_LINE
if ( prev . data > curr . data ) : NEW_LINE INDENT prev . data , curr . data = curr . data , prev . data NEW_LINE DEDENT
if ( curr . next and curr . next . data > curr . data ) : NEW_LINE INDENT curr . next . data , curr . data = curr . data , curr . next . data NEW_LINE DEDENT prev = curr . next NEW_LINE if ( not curr . next ) : NEW_LINE INDENT break NEW_LINE DEDENT curr = curr . next . next NEW_LINE return head NEW_LINE
def push ( head , k ) : NEW_LINE INDENT tem = Node ( k ) NEW_LINE tem . data = k NEW_LINE tem . next = head NEW_LINE head = tem NEW_LINE return head NEW_LINE DEDENT
def display ( head ) : NEW_LINE INDENT curr = head NEW_LINE while ( curr != None ) : NEW_LINE INDENT print ( curr . data , end = " ▁ " ) NEW_LINE curr = curr . next NEW_LINE DEDENT DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT head = None NEW_LINE DEDENT
head = push ( head , 7 ) NEW_LINE head = push ( head , 3 ) NEW_LINE head = push ( head , 8 ) NEW_LINE head = push ( head , 6 ) NEW_LINE head = push ( head , 9 ) NEW_LINE head = rearrange ( head ) NEW_LINE display ( head ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , key ) : NEW_LINE INDENT self . data = key NEW_LINE self . next = None NEW_LINE DEDENT DEDENT
left = None NEW_LINE
def printlist ( head ) : NEW_LINE INDENT while ( head != None ) : NEW_LINE INDENT print ( head . data , end = " ▁ " ) NEW_LINE if ( head . next != None ) : NEW_LINE INDENT print ( " - > " , end = " " ) NEW_LINE DEDENT head = head . next NEW_LINE DEDENT print ( ) NEW_LINE DEDENT
def rearrange ( head ) : NEW_LINE INDENT global left NEW_LINE if ( head != None ) : NEW_LINE INDENT left = head NEW_LINE reorderListUtil ( left ) NEW_LINE DEDENT DEDENT def reorderListUtil ( right ) : NEW_LINE INDENT global left NEW_LINE if ( right == None ) : NEW_LINE INDENT return NEW_LINE DEDENT reorderListUtil ( right . next ) NEW_LINE DEDENT
if ( left == None ) : NEW_LINE INDENT return NEW_LINE DEDENT
if ( left != right and left . next != right ) : NEW_LINE INDENT temp = left . next NEW_LINE left . next = right NEW_LINE right . next = temp NEW_LINE left = temp NEW_LINE DEDENT else : NEW_LINE
if ( left . next == right ) : NEW_LINE
left . next . next = None NEW_LINE left = None NEW_LINE else : NEW_LINE
left . next = None NEW_LINE left = None NEW_LINE
head = Node ( 1 ) NEW_LINE head . next = Node ( 2 ) NEW_LINE head . next . next = Node ( 3 ) NEW_LINE head . next . next . next = Node ( 4 ) NEW_LINE head . next . next . next . next = Node ( 5 ) NEW_LINE
printlist ( head ) NEW_LINE
rearrange ( head ) NEW_LINE
printlist ( head ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , new_data ) : NEW_LINE INDENT self . data = new_data NEW_LINE self . next = None NEW_LINE DEDENT DEDENT def newNode ( data ) : NEW_LINE INDENT temp = Node ( 0 ) NEW_LINE temp . data = data NEW_LINE temp . next = None NEW_LINE return temp NEW_LINE DEDENT
def getLength ( Node ) : NEW_LINE INDENT size = 0 NEW_LINE while ( Node != None ) : NEW_LINE INDENT Node = Node . next NEW_LINE size = size + 1 NEW_LINE DEDENT return size NEW_LINE DEDENT
def paddZeros ( sNode , diff ) : NEW_LINE INDENT if ( sNode == None ) : NEW_LINE INDENT return None NEW_LINE DEDENT zHead = newNode ( 0 ) NEW_LINE diff = diff - 1 NEW_LINE temp = zHead NEW_LINE while ( diff > 0 ) : NEW_LINE INDENT diff = diff - 1 NEW_LINE temp . next = newNode ( 0 ) NEW_LINE temp = temp . next NEW_LINE DEDENT temp . next = sNode NEW_LINE return zHead NEW_LINE DEDENT borrow = True NEW_LINE
def subtractLinkedListHelper ( l1 , l2 ) : NEW_LINE INDENT global borrow NEW_LINE if ( l1 == None and l2 == None and not borrow ) : NEW_LINE INDENT return None NEW_LINE DEDENT l3 = None NEW_LINE l4 = None NEW_LINE if ( l1 != None ) : NEW_LINE INDENT l3 = l1 . next NEW_LINE DEDENT if ( l2 != None ) : NEW_LINE INDENT l4 = l2 . next NEW_LINE DEDENT previous = subtractLinkedListHelper ( l3 , l4 ) NEW_LINE d1 = l1 . data NEW_LINE d2 = l2 . data NEW_LINE sub = 0 NEW_LINE DEDENT
if ( borrow ) : NEW_LINE INDENT d1 = d1 - 1 NEW_LINE borrow = False NEW_LINE DEDENT
if ( d1 < d2 ) : NEW_LINE INDENT borrow = True NEW_LINE d1 = d1 + 10 NEW_LINE DEDENT
sub = d1 - d2 NEW_LINE
current = newNode ( sub ) NEW_LINE
current . next = previous NEW_LINE return current NEW_LINE
def subtractLinkedList ( l1 , l2 ) : NEW_LINE
if ( l1 == None and l2 == None ) : NEW_LINE INDENT return None NEW_LINE DEDENT
len1 = getLength ( l1 ) NEW_LINE len2 = getLength ( l2 ) NEW_LINE lNode = None NEW_LINE sNode = None NEW_LINE temp1 = l1 NEW_LINE temp2 = l2 NEW_LINE
if ( len1 != len2 ) : NEW_LINE INDENT if ( len1 > len2 ) : NEW_LINE INDENT lNode = l1 NEW_LINE DEDENT else : NEW_LINE INDENT lNode = l2 NEW_LINE DEDENT if ( len1 > len2 ) : NEW_LINE INDENT sNode = l2 NEW_LINE DEDENT else : NEW_LINE INDENT sNode = l1 NEW_LINE DEDENT sNode = paddZeros ( sNode , abs ( len1 - len2 ) ) NEW_LINE DEDENT else : NEW_LINE
while ( l1 != None and l2 != None ) : NEW_LINE INDENT if ( l1 . data != l2 . data ) : NEW_LINE INDENT if ( l1 . data > l2 . data ) : NEW_LINE INDENT lNode = temp1 NEW_LINE DEDENT else : NEW_LINE INDENT lNode = temp2 NEW_LINE DEDENT if ( l1 . data > l2 . data ) : NEW_LINE INDENT sNode = temp2 NEW_LINE DEDENT else : NEW_LINE INDENT sNode = temp1 NEW_LINE DEDENT break NEW_LINE DEDENT l1 = l1 . next NEW_LINE l2 = l2 . next NEW_LINE DEDENT global borrow NEW_LINE
borrow = False NEW_LINE return subtractLinkedListHelper ( lNode , sNode ) NEW_LINE
def printList ( Node ) : NEW_LINE INDENT while ( Node != None ) : NEW_LINE INDENT print ( Node . data , end = " ▁ " ) NEW_LINE Node = Node . next NEW_LINE DEDENT print ( " ▁ " ) NEW_LINE DEDENT
head1 = newNode ( 1 ) NEW_LINE head1 . next = newNode ( 0 ) NEW_LINE head1 . next . next = newNode ( 0 ) NEW_LINE head2 = newNode ( 1 ) NEW_LINE result = subtractLinkedList ( head1 , head2 ) NEW_LINE printList ( result ) NEW_LINE
class Node : NEW_LINE
def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . next = None NEW_LINE DEDENT
def insertAtMid ( head , x ) : NEW_LINE
if ( head == None ) : NEW_LINE INDENT head = Node ( x ) NEW_LINE DEDENT else : NEW_LINE
newNode = Node ( x ) NEW_LINE ptr = head NEW_LINE length = 0 NEW_LINE
while ( ptr != None ) : NEW_LINE INDENT ptr = ptr . next NEW_LINE length += 1 NEW_LINE DEDENT
if ( length % 2 == 0 ) : NEW_LINE INDENT count = length / 2 NEW_LINE DEDENT else : NEW_LINE INDENT ( length + 1 ) / 2 NEW_LINE DEDENT ptr = head NEW_LINE
while ( count > 1 ) : NEW_LINE INDENT count -= 1 NEW_LINE ptr = ptr . next NEW_LINE DEDENT
newNode . next = ptr . next NEW_LINE ptr . next = newNode NEW_LINE
def display ( head ) : NEW_LINE INDENT temp = head NEW_LINE while ( temp != None ) : NEW_LINE INDENT print ( str ( temp . data ) , end = " ▁ " ) NEW_LINE temp = temp . next NEW_LINE DEDENT DEDENT
head = Node ( 1 ) NEW_LINE head . next = Node ( 2 ) NEW_LINE head . next . next = Node ( 4 ) NEW_LINE head . next . next . next = Node ( 5 ) NEW_LINE print ( " Linked ▁ list ▁ before ▁ insertion : ▁ " , end = " " ) NEW_LINE display ( head ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . prev = None NEW_LINE self . next = None NEW_LINE DEDENT DEDENT
def getNode ( data ) : NEW_LINE
newNode = Node ( 0 ) NEW_LINE
newNode . data = data NEW_LINE newNode . prev = newNode . next = None NEW_LINE return newNode NEW_LINE
def sortedInsert ( head_ref , newNode ) : NEW_LINE INDENT current = None NEW_LINE DEDENT
if ( head_ref == None ) : NEW_LINE INDENT head_ref = newNode NEW_LINE DEDENT
elif ( ( head_ref ) . data >= newNode . data ) : NEW_LINE INDENT newNode . next = head_ref NEW_LINE newNode . next . prev = newNode NEW_LINE head_ref = newNode NEW_LINE DEDENT else : NEW_LINE INDENT current = head_ref NEW_LINE DEDENT
while ( current . next != None and current . next . data < newNode . data ) : NEW_LINE INDENT current = current . next NEW_LINE DEDENT
newNode . next = current . next NEW_LINE
if ( current . next != None ) : NEW_LINE INDENT newNode . next . prev = newNode NEW_LINE DEDENT current . next = newNode NEW_LINE newNode . prev = current NEW_LINE return head_ref ; NEW_LINE
def insertionSort ( head_ref ) : NEW_LINE
sorted = None NEW_LINE
current = head_ref NEW_LINE while ( current != None ) : NEW_LINE
next = current . next NEW_LINE
current . prev = current . next = None NEW_LINE
sorted = sortedInsert ( sorted , current ) NEW_LINE
current = next NEW_LINE
head_ref = sorted NEW_LINE return head_ref NEW_LINE
def printList ( head ) : NEW_LINE INDENT while ( head != None ) : NEW_LINE INDENT print ( head . data , end = " ▁ " ) NEW_LINE head = head . next NEW_LINE DEDENT DEDENT
def push ( head_ref , new_data ) : NEW_LINE
new_node = Node ( 0 ) NEW_LINE
new_node . data = new_data NEW_LINE
new_node . next = ( head_ref ) NEW_LINE new_node . prev = None NEW_LINE
if ( ( head_ref ) != None ) : NEW_LINE INDENT ( head_ref ) . prev = new_node NEW_LINE DEDENT
( head_ref ) = new_node NEW_LINE return head_ref NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
head = None NEW_LINE
head = push ( head , 9 ) NEW_LINE head = push ( head , 3 ) NEW_LINE head = push ( head , 5 ) NEW_LINE head = push ( head , 10 ) NEW_LINE head = push ( head , 12 ) NEW_LINE head = push ( head , 8 ) NEW_LINE print ( " Doubly ▁ Linked ▁ List ▁ Before ▁ Sorting " ) NEW_LINE printList ( head ) NEW_LINE head = insertionSort ( head ) NEW_LINE print ( " Doubly Linked List After Sorting " ) NEW_LINE printList ( head ) NEW_LINE
def reverse ( arr , s , e ) : NEW_LINE INDENT while s < e : NEW_LINE INDENT tem = arr [ s ] NEW_LINE arr [ s ] = arr [ e ] NEW_LINE arr [ e ] = tem NEW_LINE s = s + 1 NEW_LINE e = e - 1 NEW_LINE DEDENT DEDENT
def fun ( arr , k ) : NEW_LINE INDENT n = len ( arr ) - 1 NEW_LINE v = n - k NEW_LINE if v >= 0 : NEW_LINE INDENT reverse ( arr , 0 , v ) NEW_LINE reverse ( arr , v + 1 , n ) NEW_LINE reverse ( arr , 0 , n ) NEW_LINE return arr NEW_LINE DEDENT DEDENT
arr = [ 1 , 2 , 3 , 4 ] NEW_LINE for i in range ( 0 , len ( arr ) ) : NEW_LINE INDENT count = 0 NEW_LINE p = fun ( arr , i ) NEW_LINE print ( p , end = " ▁ " ) NEW_LINE DEDENT
MAX = 100005 NEW_LINE
seg = [ 0 ] * ( 4 * MAX ) NEW_LINE
def build ( node , l , r , a ) : NEW_LINE INDENT if ( l == r ) : NEW_LINE INDENT seg [ node ] = a [ l ] NEW_LINE DEDENT else : NEW_LINE INDENT mid = ( l + r ) // 2 NEW_LINE build ( 2 * node , l , mid , a ) NEW_LINE build ( 2 * node + 1 , mid + 1 , r , a ) NEW_LINE seg [ node ] = ( seg [ 2 * node ] seg [ 2 * node + 1 ] ) NEW_LINE DEDENT DEDENT
def query ( node , l , r , start , end , a ) : NEW_LINE
if ( l > end or r < start ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT if ( start <= l and r <= end ) : NEW_LINE INDENT return seg [ node ] NEW_LINE DEDENT
mid = ( l + r ) // 2 NEW_LINE
return ( ( query ( 2 * node , l , mid , start , end , a ) ) | ( query ( 2 * node + 1 , mid + 1 , r , start , end , a ) ) ) NEW_LINE
def orsum ( a , n , q , k ) : NEW_LINE
build ( 1 , 0 , n - 1 , a ) NEW_LINE
for j in range ( q ) : NEW_LINE
i = k [ j ] % ( n // 2 ) NEW_LINE
sec = query ( 1 , 0 , n - 1 , n // 2 - i , n - i - 1 , a ) NEW_LINE
first = ( query ( 1 , 0 , n - 1 , 0 , n // 2 - 1 - i , a ) | query ( 1 , 0 , n - 1 , n - i , n - 1 , a ) ) NEW_LINE temp = sec + first NEW_LINE
print ( temp ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = [ 7 , 44 , 19 , 86 , 65 , 39 , 75 , 101 ] NEW_LINE n = len ( a ) NEW_LINE q = 2 NEW_LINE k = [ 4 , 2 ] NEW_LINE orsum ( a , n , q , k ) NEW_LINE DEDENT
def maximumEqual ( a , b , n ) : NEW_LINE
store = [ 0 ] * 10 ** 5 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT store [ b [ i ] ] = i + 1 NEW_LINE DEDENT
ans = [ 0 ] * 10 ** 5 NEW_LINE
for i in range ( n ) : NEW_LINE
d = abs ( store [ a [ i ] ] - ( i + 1 ) ) NEW_LINE
if ( store [ a [ i ] ] < i + 1 ) : NEW_LINE INDENT d = n - d NEW_LINE DEDENT
ans [ d ] += 1 NEW_LINE finalans = 0 NEW_LINE
for i in range ( 10 ** 5 ) : NEW_LINE INDENT finalans = max ( finalans , ans [ i ] ) NEW_LINE DEDENT
print ( finalans ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
A = [ 6 , 7 , 3 , 9 , 5 ] NEW_LINE B = [ 7 , 3 , 9 , 5 , 6 ] NEW_LINE size = len ( A ) NEW_LINE
maximumEqual ( A , B , size ) NEW_LINE
def RightRotate ( a , n , k ) : NEW_LINE
k = k % n ; NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT if ( i < k ) : NEW_LINE DEDENT
print ( a [ n + i - k ] , end = " ▁ " ) ; NEW_LINE else : NEW_LINE
print ( a [ i - k ] , end = " ▁ " ) ; NEW_LINE print ( " " ) ; NEW_LINE
Array = [ 1 , 2 , 3 , 4 , 5 ] ; NEW_LINE N = len ( Array ) ; NEW_LINE K = 2 ; NEW_LINE RightRotate ( Array , N , K ) ; NEW_LINE
def restoreSortedArray ( arr , n ) : NEW_LINE INDENT for i in range ( n ) : NEW_LINE INDENT if ( arr [ i ] > arr [ i + 1 ] ) : NEW_LINE DEDENT DEDENT
reverse ( arr , 0 , i ) ; NEW_LINE reverse ( arr , i + 1 , n ) ; NEW_LINE reverse ( arr , 0 , n ) ; NEW_LINE def reverse ( arr , i , j ) : NEW_LINE while ( i < j ) : NEW_LINE temp = arr [ i ] ; NEW_LINE arr [ i ] = arr [ j ] ; NEW_LINE arr [ j ] = temp ; NEW_LINE i += 1 ; NEW_LINE j -= 1 ; NEW_LINE
def printArray ( arr , size ) : NEW_LINE INDENT for i in range ( size ) : NEW_LINE INDENT print ( arr [ i ] , end = " " ) ; NEW_LINE DEDENT DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 3 , 4 , 5 , 1 , 2 ] ; NEW_LINE n = len ( arr ) ; NEW_LINE restoreSortedArray ( arr , n - 1 ) ; NEW_LINE printArray ( arr , n ) ; NEW_LINE DEDENT
def findStartIndexOfArray ( arr , low , high ) : NEW_LINE INDENT if ( low > high ) : NEW_LINE INDENT return - 1 ; NEW_LINE DEDENT if ( low == high ) : NEW_LINE INDENT return low ; NEW_LINE DEDENT mid = low + ( high - low ) / 2 ; NEW_LINE if ( arr [ mid ] > arr [ mid + 1 ] ) : NEW_LINE INDENT return mid + 1 ; NEW_LINE DEDENT if ( arr [ mid - 1 ] > arr [ mid ] ) : NEW_LINE INDENT return mid ; NEW_LINE DEDENT if ( arr [ low ] > arr [ mid ] ) : NEW_LINE INDENT return findStartIndexOfArray ( arr , low , mid - 1 ) ; NEW_LINE DEDENT else : NEW_LINE INDENT return findStartIndexOfArray ( arr , mid + 1 , high ) ; NEW_LINE DEDENT DEDENT
def restoreSortedArray ( arr , n ) : NEW_LINE
if ( arr [ 0 ] < arr [ n - 1 ] ) : NEW_LINE INDENT return ; NEW_LINE DEDENT start = findStartIndexOfArray ( arr , 0 , n - 1 ) ; NEW_LINE
reverse ( arr , 0 , start ) ; NEW_LINE reverse ( arr , start , n ) ; NEW_LINE reverse ( arr ) ; NEW_LINE
def printArray ( arr , size ) : NEW_LINE INDENT for i in range ( size ) : NEW_LINE INDENT print ( arr [ i ] , end = " " ) ; NEW_LINE DEDENT DEDENT def reverse ( arr , i , j ) : NEW_LINE INDENT while ( i < j ) : NEW_LINE INDENT temp = arr [ i ] ; NEW_LINE arr [ i ] = arr [ j ] ; NEW_LINE arr [ j ] = temp ; NEW_LINE i += 1 ; NEW_LINE j -= 1 ; NEW_LINE DEDENT DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 1 , 2 , 3 , 4 , 5 ] ; NEW_LINE n = len ( arr ) ; NEW_LINE restoreSortedArray ( arr , n ) ; NEW_LINE printArray ( arr , n ) ; NEW_LINE DEDENT
def leftrotate ( s , d ) : NEW_LINE INDENT tmp = s [ d : ] + s [ 0 : d ] NEW_LINE return tmp NEW_LINE DEDENT
def rightrotate ( s , d ) : NEW_LINE INDENT return leftrotate ( s , len ( s ) - d ) NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT str1 = " GeeksforGeeks " NEW_LINE print ( leftrotate ( str1 , 2 ) ) NEW_LINE str2 = " GeeksforGeeks " NEW_LINE print ( rightrotate ( str2 , 2 ) ) NEW_LINE DEDENT
import math NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . next = None NEW_LINE DEDENT DEDENT
def insertNode ( start , value ) : NEW_LINE
if ( start == None ) : NEW_LINE INDENT new_node = Node ( value ) NEW_LINE new_node . data = value NEW_LINE new_node . next = new_node NEW_LINE new_node . prev = new_node NEW_LINE start = new_node NEW_LINE return new_node NEW_LINE DEDENT
last = start . prev NEW_LINE
new_node = Node ( value ) NEW_LINE new_node . data = value NEW_LINE
new_node . next = start NEW_LINE
( start ) . prev = new_node NEW_LINE
new_node . prev = last NEW_LINE
last . next = new_node NEW_LINE return start NEW_LINE
def displayList ( start ) : NEW_LINE INDENT temp = start NEW_LINE while ( temp . next != start ) : NEW_LINE INDENT print ( temp . data , end = " ▁ " ) NEW_LINE temp = temp . next NEW_LINE DEDENT print ( temp . data ) NEW_LINE DEDENT
def searchList ( start , search ) : NEW_LINE
temp = start NEW_LINE
count = 0 NEW_LINE flag = 0 NEW_LINE value = 0 NEW_LINE
if ( temp == None ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT else : NEW_LINE
while ( temp . next != start ) : NEW_LINE
count = count + 1 NEW_LINE
if ( temp . data == search ) : NEW_LINE INDENT flag = 1 NEW_LINE count = count - 1 NEW_LINE break NEW_LINE DEDENT
temp = temp . next NEW_LINE
if ( temp . data == search ) : NEW_LINE INDENT count = count + 1 NEW_LINE flag = 1 NEW_LINE DEDENT
if ( flag == 1 ) : NEW_LINE INDENT print ( search , " found ▁ at ▁ location ▁ " , count ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( search , " ▁ not ▁ found " ) NEW_LINE DEDENT return - 1 NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
start = None NEW_LINE
start = insertNode ( start , 4 ) NEW_LINE
start = insertNode ( start , 5 ) NEW_LINE
start = insertNode ( start , 7 ) NEW_LINE
start = insertNode ( start , 8 ) NEW_LINE
start = insertNode ( start , 6 ) NEW_LINE print ( " Created ▁ circular ▁ doubly ▁ linked ▁ list ▁ is : ▁ " , end = " " ) NEW_LINE displayList ( start ) NEW_LINE searchList ( start , 5 ) NEW_LINE
import math NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . next = None NEW_LINE DEDENT DEDENT
def getNode ( data ) : NEW_LINE INDENT newNode = Node ( data ) NEW_LINE newNode . data = data NEW_LINE return newNode NEW_LINE DEDENT
def insertEnd ( head , new_node ) : NEW_LINE
if ( head == None ) : NEW_LINE INDENT new_node . next = new_node NEW_LINE new_node . prev = new_node NEW_LINE head = new_node NEW_LINE return head NEW_LINE DEDENT
last = head . prev NEW_LINE
new_node . next = head NEW_LINE
head . prev = new_node NEW_LINE
new_node . prev = last NEW_LINE
last . next = new_node NEW_LINE return head NEW_LINE
def reverse ( head ) : NEW_LINE INDENT if ( head == None ) : NEW_LINE INDENT return None NEW_LINE DEDENT DEDENT
new_head = None NEW_LINE
last = head . prev NEW_LINE
curr = last NEW_LINE
while ( curr . prev != last ) : NEW_LINE INDENT prev = curr . prev NEW_LINE DEDENT
new_head = insertEnd ( new_head , curr ) NEW_LINE curr = prev NEW_LINE new_head = insertEnd ( new_head , curr ) NEW_LINE
return new_head NEW_LINE
def display ( head ) : NEW_LINE INDENT if ( head == None ) : NEW_LINE INDENT return NEW_LINE DEDENT temp = head NEW_LINE print ( " Forward ▁ direction : ▁ " , end = " " ) NEW_LINE while ( temp . next != head ) : NEW_LINE INDENT print ( temp . data , end = " ▁ " ) NEW_LINE temp = temp . next NEW_LINE DEDENT print ( temp . data ) NEW_LINE last = head . prev NEW_LINE temp = last NEW_LINE print ( " Backward ▁ direction : ▁ " , end = " " ) NEW_LINE while ( temp . prev != last ) : NEW_LINE INDENT print ( temp . data , end = " ▁ " ) NEW_LINE temp = temp . prev NEW_LINE DEDENT print ( temp . data ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT head = None NEW_LINE head = insertEnd ( head , getNode ( 1 ) ) NEW_LINE head = insertEnd ( head , getNode ( 2 ) ) NEW_LINE head = insertEnd ( head , getNode ( 3 ) ) NEW_LINE head = insertEnd ( head , getNode ( 4 ) ) NEW_LINE head = insertEnd ( head , getNode ( 5 ) ) NEW_LINE print ( " Current ▁ list : " ) NEW_LINE display ( head ) NEW_LINE head = reverse ( head ) NEW_LINE print ( " Reversed list : " ) NEW_LINE display ( head ) NEW_LINE DEDENT
MAXN = 1001 NEW_LINE
depth = [ 0 for i in range ( MAXN ) ] ; NEW_LINE
parent = [ 0 for i in range ( MAXN ) ] ; NEW_LINE adj = [ [ ] for i in range ( MAXN ) ] NEW_LINE def addEdge ( u , v ) : NEW_LINE INDENT adj [ u ] . append ( v ) ; NEW_LINE adj [ v ] . append ( u ) ; NEW_LINE DEDENT def dfs ( cur , prev ) : NEW_LINE
parent [ cur ] = prev ; NEW_LINE
depth [ cur ] = depth [ prev ] + 1 ; NEW_LINE
for i in range ( len ( adj [ cur ] ) ) : NEW_LINE INDENT if ( adj [ cur ] [ i ] != prev ) : NEW_LINE INDENT dfs ( adj [ cur ] [ i ] , cur ) ; NEW_LINE DEDENT DEDENT def preprocess ( ) : NEW_LINE
depth [ 0 ] = - 1 ; NEW_LINE
dfs ( 1 , 0 ) ; NEW_LINE
def LCANaive ( u , v ) : NEW_LINE INDENT if ( u == v ) : NEW_LINE INDENT return u ; NEW_LINE DEDENT if ( depth [ u ] > depth [ v ] ) : NEW_LINE INDENT u , v = v , u NEW_LINE DEDENT v = parent [ v ] ; NEW_LINE return LCANaive ( u , v ) ; NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE
addEdge ( 1 , 2 ) ; NEW_LINE addEdge ( 1 , 3 ) ; NEW_LINE addEdge ( 1 , 4 ) ; NEW_LINE addEdge ( 2 , 5 ) ; NEW_LINE addEdge ( 2 , 6 ) ; NEW_LINE addEdge ( 3 , 7 ) ; NEW_LINE addEdge ( 4 , 8 ) ; NEW_LINE addEdge ( 4 , 9 ) ; NEW_LINE addEdge ( 9 , 10 ) ; NEW_LINE addEdge ( 9 , 11 ) ; NEW_LINE addEdge ( 7 , 12 ) ; NEW_LINE addEdge ( 7 , 13 ) ; NEW_LINE preprocess ( ) ; NEW_LINE print ( ' LCA ( 11,8 ) ▁ : ▁ ' + str ( LCANaive ( 11 , 8 ) ) ) NEW_LINE print ( ' LCA ( 3,13 ) ▁ : ▁ ' + str ( LCANaive ( 3 , 13 ) ) ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 3 NEW_LINE DEDENT
print ( pow ( 2 , N + 1 ) - 2 ) NEW_LINE
def countOfNum ( n , a , b ) : NEW_LINE INDENT cnt_of_a , cnt_of_b , cnt_of_ab , sum = 0 , 0 , 0 , 0 NEW_LINE DEDENT
cnt_of_a = n // a NEW_LINE
cnt_of_b = n // b NEW_LINE
sum = cnt_of_b + cnt_of_a NEW_LINE
cnt_of_ab = n // ( a * b ) NEW_LINE
sum = sum - cnt_of_ab NEW_LINE return sum NEW_LINE
def sumOfNum ( n , a , b ) : NEW_LINE INDENT i = 0 NEW_LINE sum = 0 NEW_LINE DEDENT
ans = dict ( ) NEW_LINE
for i in range ( a , n + 1 , a ) : NEW_LINE INDENT ans [ i ] = 1 NEW_LINE DEDENT
for i in range ( b , n + 1 , b ) : NEW_LINE INDENT ans [ i ] = 1 NEW_LINE DEDENT
for it in ans : NEW_LINE INDENT sum = sum + it NEW_LINE DEDENT return sum NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 88 NEW_LINE A = 11 NEW_LINE B = 8 NEW_LINE count = countOfNum ( N , A , B ) NEW_LINE sumofnum = sumOfNum ( N , A , B ) NEW_LINE print ( sumofnum % count ) NEW_LINE DEDENT
def get ( L , R ) : NEW_LINE
x = 1.0 / L ; NEW_LINE
y = 1.0 / ( R + 1.0 ) ; NEW_LINE return ( x - y ) ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT L = 6 ; R = 12 ; NEW_LINE DEDENT
ans = get ( L , R ) ; NEW_LINE print ( round ( ans , 2 ) ) ; NEW_LINE
from bisect import bisect_right as upper_bound NEW_LINE MAX = 100000 NEW_LINE
v = [ ] NEW_LINE
def consecutiveOnes ( x ) : NEW_LINE
p = 0 NEW_LINE while ( x > 0 ) : NEW_LINE
if ( x % 2 == 1 and p == 1 ) : NEW_LINE INDENT return True NEW_LINE DEDENT
p = x % 2 NEW_LINE
x //= 2 NEW_LINE return False NEW_LINE
def preCompute ( ) : NEW_LINE
for i in range ( MAX + 1 ) : NEW_LINE INDENT if ( consecutiveOnes ( i ) == 0 ) : NEW_LINE INDENT v . append ( i ) NEW_LINE DEDENT DEDENT
def nextValid ( n ) : NEW_LINE
it = upper_bound ( v , n ) NEW_LINE val = v [ it ] NEW_LINE return val NEW_LINE
def performQueries ( queries , q ) : NEW_LINE INDENT for i in range ( q ) : NEW_LINE INDENT print ( nextValid ( queries [ i ] ) ) NEW_LINE DEDENT DEDENT
queries = [ 4 , 6 ] NEW_LINE q = len ( queries ) NEW_LINE
preCompute ( ) NEW_LINE
performQueries ( queries , q ) NEW_LINE
def changeToOnes ( string ) : NEW_LINE
ctr = 0 ; NEW_LINE l = len ( string ) ; NEW_LINE
for i in range ( l - 1 , - 1 , - 1 ) : NEW_LINE
if ( string [ i ] == '1' ) : NEW_LINE INDENT ctr += 1 ; NEW_LINE DEDENT
else : NEW_LINE INDENT break ; NEW_LINE DEDENT
return l - ctr ; NEW_LINE
def removeZeroesFromFront ( string ) : NEW_LINE INDENT s = " " ; NEW_LINE i = 0 ; NEW_LINE DEDENT
while ( i < len ( string ) and string [ i ] == '0' ) : NEW_LINE INDENT i += 1 ; NEW_LINE DEDENT
if ( i == len ( string ) ) : NEW_LINE INDENT s = "0" ; NEW_LINE DEDENT
else : NEW_LINE INDENT s = string [ i : len ( string ) - i ] ; NEW_LINE DEDENT return s ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT string = "10010111" ; NEW_LINE DEDENT
string = removeZeroesFromFront ( string ) ; NEW_LINE print ( changeToOnes ( string ) ) ; NEW_LINE
def MinDeletion ( a , n ) : NEW_LINE
map = dict . fromkeys ( a , 0 ) ; NEW_LINE
for i in range ( n ) : NEW_LINE INDENT map [ a [ i ] ] += 1 ; NEW_LINE DEDENT
ans = 0 ; NEW_LINE for key , value in map . items ( ) : NEW_LINE
x = key ; NEW_LINE
frequency = value ; NEW_LINE
if ( x <= frequency ) : NEW_LINE
ans += ( frequency - x ) ; NEW_LINE
else : NEW_LINE INDENT ans += frequency ; NEW_LINE DEDENT return ans ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = [ 2 , 3 , 2 , 3 , 4 , 4 , 4 , 4 , 5 ] ; NEW_LINE n = len ( a ) ; NEW_LINE print ( MinDeletion ( a , n ) ) ; NEW_LINE DEDENT
def maxCountAB ( s , n ) : NEW_LINE
A = 0 NEW_LINE B = 0 NEW_LINE BA = 0 NEW_LINE ans = 0 NEW_LINE for i in range ( n ) : NEW_LINE INDENT S = s [ i ] NEW_LINE L = len ( S ) NEW_LINE for j in range ( L - 1 ) : NEW_LINE DEDENT
if ( S [ j ] == ' A ' and S [ j + 1 ] == ' B ' ) : NEW_LINE INDENT ans += 1 NEW_LINE DEDENT
if ( S [ 0 ] == ' B ' and S [ L - 1 ] == ' A ' ) : NEW_LINE INDENT BA += 1 NEW_LINE DEDENT
' NEW_LINE INDENT elif ( S [ 0 ] == ' B ' ) : NEW_LINE INDENT B += 1 NEW_LINE DEDENT DEDENT
' NEW_LINE INDENT elif ( S [ L - 1 ] == ' A ' ) : NEW_LINE INDENT A += 1 NEW_LINE DEDENT DEDENT
' NEW_LINE INDENT if ( BA == 0 ) : NEW_LINE INDENT ans += min ( B , A ) NEW_LINE DEDENT elif ( A + B == 0 ) : NEW_LINE INDENT ans += BA - 1 NEW_LINE DEDENT else : NEW_LINE INDENT ans += BA + min ( B , A ) NEW_LINE DEDENT return ans NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT s = [ " ABCA " , " BOOK " , " BAND " ] NEW_LINE n = len ( s ) NEW_LINE print ( maxCountAB ( s , n ) ) NEW_LINE DEDENT
def MinOperations ( n , x , arr ) : NEW_LINE
total = 0 NEW_LINE for i in range ( n ) : NEW_LINE
if ( arr [ i ] > x ) : NEW_LINE INDENT difference = arr [ i ] - x NEW_LINE total = total + difference NEW_LINE arr [ i ] = x NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT LeftNeigbouringSum = arr [ i ] + arr [ i - 1 ] NEW_LINE DEDENT
if ( LeftNeigbouringSum > x ) : NEW_LINE INDENT current_diff = LeftNeigbouringSum - x NEW_LINE arr [ i ] = max ( 0 , arr [ i ] - current_diff ) NEW_LINE total = total + current_diff NEW_LINE DEDENT return total NEW_LINE
X = 1 NEW_LINE arr = [ 1 , 6 , 1 , 2 , 0 , 4 ] NEW_LINE N = len ( arr ) NEW_LINE print ( MinOperations ( N , X , arr ) ) NEW_LINE
import math NEW_LINE
def findNumbers ( arr , n ) : NEW_LINE
sumN = ( n * ( n + 1 ) ) / 2 ; NEW_LINE
sumSqN = ( n * ( n + 1 ) * ( 2 * n + 1 ) ) / 6 ; NEW_LINE
sum = 0 ; NEW_LINE sumSq = 0 ; NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT sum = sum + arr [ i ] ; NEW_LINE sumSq = sumSq + ( math . pow ( arr [ i ] , 2 ) ) ; NEW_LINE DEDENT B = ( ( ( sumSq - sumSqN ) / ( sum - sumN ) ) + sumN - sum ) / 2 ; NEW_LINE A = sum - sumN + B ; NEW_LINE print ( " A ▁ = ▁ " , int ( A ) ) ; NEW_LINE print ( " B ▁ = ▁ " , int ( B ) ) ; NEW_LINE
arr = [ 1 , 2 , 2 , 3 , 4 ] ; NEW_LINE n = len ( arr ) ; NEW_LINE findNumbers ( arr , n ) ; NEW_LINE
def is_prefix ( temp , str ) : NEW_LINE
if ( len ( temp ) < len ( str ) ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT else : NEW_LINE
for i in range ( len ( str ) ) : NEW_LINE INDENT if ( str [ i ] != temp [ i ] ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT DEDENT return 1 NEW_LINE
def lexicographicallyString ( input , n , str ) : NEW_LINE
input . sort ( ) NEW_LINE for i in range ( n ) : NEW_LINE INDENT temp = input [ i ] NEW_LINE DEDENT
if ( is_prefix ( temp , str ) ) : NEW_LINE INDENT return temp NEW_LINE DEDENT
return " - 1" NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ " apple " , " appe " , " apl " , " aapl " , " appax " ] NEW_LINE S = " app " NEW_LINE N = 5 NEW_LINE print ( lexicographicallyString ( arr , N , S ) ) NEW_LINE DEDENT
def Rearrange ( arr , K , N ) : NEW_LINE
ans = [ 0 ] * ( N + 1 ) NEW_LINE
f = - 1 NEW_LINE for i in range ( N ) : NEW_LINE INDENT ans [ i ] = - 1 NEW_LINE DEDENT
K = arr . index ( K ) NEW_LINE
smaller = [ ] NEW_LINE greater = [ ] NEW_LINE
for i in range ( N ) : NEW_LINE
if ( arr [ i ] < arr [ K ] ) : NEW_LINE INDENT smaller . append ( arr [ i ] ) NEW_LINE DEDENT
elif ( arr [ i ] > arr [ K ] ) : NEW_LINE INDENT greater . append ( arr [ i ] ) NEW_LINE DEDENT low = 0 NEW_LINE high = N - 1 NEW_LINE
while ( low <= high ) : NEW_LINE
mid = ( low + high ) // 2 NEW_LINE
if ( mid == K ) : NEW_LINE INDENT ans [ mid ] = arr [ K ] NEW_LINE f = 1 NEW_LINE break NEW_LINE DEDENT
elif ( mid < K ) : NEW_LINE INDENT if ( len ( smaller ) == 0 ) : NEW_LINE INDENT break NEW_LINE DEDENT ans [ mid ] = smaller [ - 1 ] NEW_LINE smaller . pop ( ) NEW_LINE low = mid + 1 NEW_LINE DEDENT
else : NEW_LINE INDENT if ( len ( greater ) == 0 ) : NEW_LINE INDENT break NEW_LINE DEDENT ans [ mid ] = greater [ - 1 ] NEW_LINE greater . pop ( ) NEW_LINE high = mid - 1 NEW_LINE DEDENT
if ( f == - 1 ) : NEW_LINE INDENT print ( - 1 ) NEW_LINE return NEW_LINE DEDENT
for i in range ( N ) : NEW_LINE
if ( ans [ i ] == - 1 ) : NEW_LINE INDENT if ( len ( smaller ) ) : NEW_LINE INDENT ans [ i ] = smaller [ - 1 ] NEW_LINE smaller . pop ( ) NEW_LINE DEDENT elif ( len ( greater ) ) : NEW_LINE INDENT ans [ i ] = greater [ - 1 ] NEW_LINE greater . pop ( ) NEW_LINE DEDENT DEDENT
for i in range ( N ) : NEW_LINE INDENT print ( ans [ i ] , end = " ▁ " ) NEW_LINE DEDENT print ( ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
arr = [ 10 , 7 , 2 , 5 , 3 , 8 ] NEW_LINE K = 7 NEW_LINE N = len ( arr ) NEW_LINE
Rearrange ( arr , K , N ) NEW_LINE
import math NEW_LINE
def minimumK ( arr , M , N ) : NEW_LINE
good = math . ceil ( ( N * 1.0 ) / ( ( M + 1 ) * 1.0 ) ) NEW_LINE
for i in range ( 1 , N + 1 ) : NEW_LINE INDENT K = i NEW_LINE DEDENT
candies = N NEW_LINE
taken = 0 NEW_LINE while ( candies > 0 ) : NEW_LINE
taken += min ( K , candies ) NEW_LINE candies -= min ( K , candies ) NEW_LINE
for j in range ( M ) : NEW_LINE
consume = ( arr [ j ] * candies ) / 100 NEW_LINE
candies -= consume NEW_LINE
if ( taken >= good ) : NEW_LINE print ( i ) NEW_LINE return NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT N = 13 NEW_LINE M = 1 NEW_LINE arr = [ 50 ] NEW_LINE minimumK ( arr , M , N ) NEW_LINE DEDENT
def calcTotalTime ( path ) : NEW_LINE
time = 0 NEW_LINE
x = 0 NEW_LINE y = 0 NEW_LINE
s = set ( [ ] ) NEW_LINE for i in range ( len ( path ) ) : NEW_LINE INDENT p = x NEW_LINE q = y NEW_LINE if ( path [ i ] == ' N ' ) : NEW_LINE INDENT y += 1 NEW_LINE DEDENT elif ( path [ i ] == ' S ' ) : NEW_LINE INDENT y -= 1 NEW_LINE DEDENT elif ( path [ i ] == ' E ' ) : NEW_LINE INDENT x += 1 NEW_LINE DEDENT elif ( path [ i ] == ' W ' ) : NEW_LINE INDENT x -= 1 NEW_LINE DEDENT DEDENT
if ( p + x , q + y ) not in s : NEW_LINE
time += 2 NEW_LINE
s . add ( ( p + x , q + y ) ) NEW_LINE else : NEW_LINE time += 1 NEW_LINE
print ( time ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT path = " NSE " NEW_LINE calcTotalTime ( path ) NEW_LINE DEDENT
def findCost ( A , N ) : NEW_LINE
totalCost = 0 NEW_LINE
for i in range ( N ) : NEW_LINE
if ( A [ i ] == 0 ) : NEW_LINE
A [ i ] = 1 NEW_LINE
totalCost += i NEW_LINE
return totalCost NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 1 , 0 , 1 , 0 , 1 , 0 ] NEW_LINE N = len ( arr ) NEW_LINE print ( findCost ( arr , N ) ) NEW_LINE DEDENT
def peakIndex ( arr ) : NEW_LINE INDENT N = len ( arr ) NEW_LINE DEDENT
if ( len ( arr ) < 3 ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT i = 0 NEW_LINE
while ( i + 1 < N ) : NEW_LINE
if ( arr [ i + 1 ] < arr [ i ] or arr [ i ] == arr [ i + 1 ] ) : NEW_LINE INDENT break NEW_LINE DEDENT i += 1 NEW_LINE if ( i == 0 or i == N - 1 ) : NEW_LINE return - 1 NEW_LINE
ans = i NEW_LINE
while ( i < N - 1 ) : NEW_LINE
if ( arr [ i ] < arr [ i + 1 ] or arr [ i ] == arr [ i + 1 ] ) : NEW_LINE INDENT break NEW_LINE DEDENT i += 1 NEW_LINE
if ( i == N - 1 ) : NEW_LINE INDENT return ans NEW_LINE DEDENT
return - 1 NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 0 , 1 , 0 ] NEW_LINE print ( peakIndex ( arr ) ) NEW_LINE DEDENT
def hasArrayTwoPairs ( nums , n , target ) : NEW_LINE
nums = sorted ( nums ) NEW_LINE
for i in range ( n ) : NEW_LINE
x = target - nums [ i ] NEW_LINE
low , high = 0 , n - 1 NEW_LINE while ( low <= high ) : NEW_LINE
mid = low + ( ( high - low ) // 2 ) NEW_LINE
if ( nums [ mid ] > x ) : NEW_LINE INDENT high = mid - 1 NEW_LINE DEDENT
elif ( nums [ mid ] < x ) : NEW_LINE INDENT low = mid + 1 NEW_LINE DEDENT
else : NEW_LINE
if ( mid == i ) : NEW_LINE INDENT if ( ( mid - 1 >= 0 ) and nums [ mid - 1 ] == x ) : NEW_LINE INDENT print ( nums [ i ] , end = " , ▁ " ) NEW_LINE print ( nums [ mid - 1 ] ) NEW_LINE return NEW_LINE DEDENT if ( ( mid + 1 < n ) and nums [ mid + 1 ] == x ) : NEW_LINE INDENT print ( nums [ i ] , end = " , ▁ " ) NEW_LINE print ( nums [ mid + 1 ] ) NEW_LINE return NEW_LINE DEDENT break NEW_LINE DEDENT
else : NEW_LINE INDENT print ( nums [ i ] , end = " , ▁ " ) NEW_LINE print ( nums [ mid ] ) NEW_LINE return NEW_LINE DEDENT
print ( - 1 ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT A = [ 0 , - 1 , 2 , - 3 , 1 ] NEW_LINE X = - 2 NEW_LINE N = len ( A ) NEW_LINE DEDENT
hasArrayTwoPairs ( A , N , X ) NEW_LINE
from math import sqrt , floor , ceil NEW_LINE
def findClosest ( N , target ) : NEW_LINE INDENT closest = - 1 NEW_LINE diff = 10 ** 18 NEW_LINE DEDENT
for i in range ( 1 , ceil ( sqrt ( N ) ) + 1 ) : NEW_LINE INDENT if ( N % i == 0 ) : NEW_LINE DEDENT
if ( N // i == i ) : NEW_LINE
if ( abs ( target - i ) < diff ) : NEW_LINE INDENT diff = abs ( target - i ) NEW_LINE closest = i NEW_LINE DEDENT else : NEW_LINE
if ( abs ( target - i ) < diff ) : NEW_LINE INDENT diff = abs ( target - i ) NEW_LINE closest = i NEW_LINE DEDENT
if ( abs ( target - N // i ) < diff ) : NEW_LINE INDENT diff = abs ( target - N // i ) NEW_LINE closest = N // i NEW_LINE DEDENT
print ( closest ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
N , X = 16 , 5 NEW_LINE
findClosest ( N , X ) NEW_LINE
def power ( A , N ) : NEW_LINE
count = 0 ; NEW_LINE if ( A == 1 ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT while ( N > 0 ) : NEW_LINE
count += 1 ; NEW_LINE
N //= A ; NEW_LINE return int ( count ) ; NEW_LINE
def Pairs ( N , A , B ) : NEW_LINE INDENT powerA , powerB = 0 , 0 ; NEW_LINE DEDENT
powerA = power ( A , N ) ; NEW_LINE
powerB = power ( B , N ) ; NEW_LINE
intialB = B ; NEW_LINE intialA = A ; NEW_LINE
A = 1 ; NEW_LINE for i in range ( powerA + 1 ) : NEW_LINE INDENT B = 1 ; NEW_LINE for j in range ( powerB + 1 ) : NEW_LINE DEDENT
if ( B == N - A ) : NEW_LINE INDENT print ( i , " ▁ " , j ) ; NEW_LINE return ; NEW_LINE DEDENT
B *= intialB ; NEW_LINE
A *= intialA ; NEW_LINE
print ( " - 1" ) ; NEW_LINE return ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
N = 106 ; NEW_LINE A = 3 ; NEW_LINE B = 5 ; NEW_LINE
Pairs ( N , A , B ) ; NEW_LINE
def findNonMultiples ( arr , n , k ) : NEW_LINE
multiples = set ( [ ] ) NEW_LINE
for i in range ( n ) : NEW_LINE
if ( arr [ i ] not in multiples ) : NEW_LINE
for j in range ( 1 , k // arr [ i ] + 1 ) : NEW_LINE INDENT multiples . add ( arr [ i ] * j ) NEW_LINE DEDENT
return k - len ( multiples ) NEW_LINE
def countValues ( arr , N , L , R ) : NEW_LINE
return ( findNonMultiples ( arr , N , R ) - findNonMultiples ( arr , N , L - 1 ) ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 2 , 3 , 4 , 5 , 6 ] NEW_LINE N = len ( arr ) NEW_LINE L = 1 NEW_LINE R = 20 NEW_LINE DEDENT
print ( countValues ( arr , N , L , R ) ) NEW_LINE
def minCollectingSpeed ( piles , H ) : NEW_LINE
ans = - 1 NEW_LINE low = 1 NEW_LINE
high = max ( piles ) NEW_LINE
while ( low <= high ) : NEW_LINE
K = low + ( high - low ) // 2 NEW_LINE time = 0 NEW_LINE
for ai in piles : NEW_LINE time += ( ai + K - 1 ) // K NEW_LINE
if ( time <= H ) : NEW_LINE INDENT ans = K NEW_LINE high = K - 1 NEW_LINE DEDENT
else : NEW_LINE INDENT low = K + 1 NEW_LINE DEDENT
print ( ans ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 3 , 6 , 7 , 11 ] NEW_LINE H = 8 NEW_LINE DEDENT
minCollectingSpeed ( arr , H ) NEW_LINE
def cntDisPairs ( arr , N , K ) : NEW_LINE
cntPairs = 0 NEW_LINE
arr = sorted ( arr ) NEW_LINE
i = 0 NEW_LINE
j = N - 1 NEW_LINE
while ( i < j ) : NEW_LINE
if ( arr [ i ] + arr [ j ] == K ) : NEW_LINE
while ( i < j and arr [ i ] == arr [ i + 1 ] ) : NEW_LINE
i += 1 NEW_LINE
while ( i < j and arr [ j ] == arr [ j - 1 ] ) : NEW_LINE
j -= 1 NEW_LINE
cntPairs += 1 NEW_LINE
i += 1 NEW_LINE
j -= 1 NEW_LINE
elif ( arr [ i ] + arr [ j ] < K ) : NEW_LINE
i += 1 NEW_LINE else : NEW_LINE
j -= 1 NEW_LINE return cntPairs NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 5 , 6 , 5 , 7 , 7 , 8 ] NEW_LINE N = len ( arr ) NEW_LINE K = 13 NEW_LINE print ( cntDisPairs ( arr , N , K ) ) NEW_LINE DEDENT
def cntDisPairs ( arr , N , K ) : NEW_LINE
cntPairs = 0 NEW_LINE
cntFre = { } NEW_LINE for i in arr : NEW_LINE
if i in cntFre : NEW_LINE INDENT cntFre [ i ] += 1 NEW_LINE DEDENT else : NEW_LINE INDENT cntFre [ i ] = 1 NEW_LINE DEDENT
for key , value in cntFre . items ( ) : NEW_LINE
i = key NEW_LINE
if ( 2 * i == K ) : NEW_LINE
if ( cntFre [ i ] > 1 ) : NEW_LINE INDENT cntPairs += 2 NEW_LINE DEDENT else : NEW_LINE if ( cntFre [ K - i ] ) : NEW_LINE
cntPairs += 1 NEW_LINE
cntPairs = cntPairs / 2 NEW_LINE return cntPairs NEW_LINE
arr = [ 5 , 6 , 5 , 7 , 7 , 8 ] NEW_LINE N = len ( arr ) NEW_LINE K = 13 NEW_LINE print ( int ( cntDisPairs ( arr , N , K ) ) ) NEW_LINE
def longestSubsequence ( N , Q , arr , Queries ) : NEW_LINE INDENT for i in range ( Q ) : NEW_LINE DEDENT
x = Queries [ i ] [ 0 ] NEW_LINE y = Queries [ i ] [ 1 ] NEW_LINE
arr [ x - 1 ] = y NEW_LINE
count = 1 NEW_LINE for j in range ( 1 , N ) : NEW_LINE
if ( arr [ j ] != arr [ j - 1 ] ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT
print ( count , end = ' ▁ ' ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 1 , 1 , 2 , 5 , 2 ] NEW_LINE N = len ( arr ) NEW_LINE Q = 2 NEW_LINE Queries = [ [ 1 , 3 ] , [ 4 , 2 ] ] NEW_LINE DEDENT
longestSubsequence ( N , Q , arr , Queries ) NEW_LINE
def longestSubsequence ( N , Q , arr , Queries ) : NEW_LINE INDENT count = 1 NEW_LINE DEDENT
for i in range ( 1 , N ) : NEW_LINE
if ( arr [ i ] != arr [ i - 1 ] ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT
for i in range ( Q ) : NEW_LINE
x = Queries [ i ] [ 0 ] NEW_LINE y = Queries [ i ] [ 1 ] NEW_LINE
if ( x > 1 ) : NEW_LINE
if ( arr [ x - 1 ] != arr [ x - 2 ] ) : NEW_LINE INDENT count -= 1 NEW_LINE DEDENT
if ( arr [ x - 2 ] != y ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT
if ( x < N ) : NEW_LINE
if ( arr [ x ] != arr [ x - 1 ] ) : NEW_LINE INDENT count -= 1 NEW_LINE DEDENT
if ( y != arr [ x ] ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT print ( count , end = ' ▁ ' ) NEW_LINE
arr [ x - 1 ] = y NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 1 , 1 , 2 , 5 , 2 ] NEW_LINE N = len ( arr ) NEW_LINE Q = 2 NEW_LINE Queries = [ [ 1 , 3 ] , [ 4 , 2 ] ] NEW_LINE DEDENT
longestSubsequence ( N , Q , arr , Queries ) NEW_LINE
from collections import defaultdict NEW_LINE
def sum_i ( arr , n ) : NEW_LINE
mp = defaultdict ( lambda : [ ] ) NEW_LINE
for i in range ( n ) : NEW_LINE INDENT mp [ arr [ i ] ] . append ( i ) NEW_LINE DEDENT
ans = [ 0 ] * n NEW_LINE
for i in range ( n ) : NEW_LINE
sum = 0 NEW_LINE
for it in mp [ arr [ i ] ] : NEW_LINE
sum += abs ( it - i ) NEW_LINE
ans [ i ] = sum NEW_LINE
for i in range ( n ) : NEW_LINE INDENT print ( ans [ i ] , end = " ▁ " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE
arr = [ 1 , 3 , 1 , 1 , 2 ] NEW_LINE
n = len ( arr ) NEW_LINE
sum_i ( arr , n ) NEW_LINE
def conVowUpp ( str ) : NEW_LINE
N = len ( str ) NEW_LINE str1 = " " NEW_LINE for i in range ( N ) : NEW_LINE INDENT if ( str [ i ] == ' a ' or str [ i ] == ' e ' or str [ i ] == ' i ' or str [ i ] == ' o ' or str [ i ] == ' u ' ) : NEW_LINE INDENT c = ( str [ i ] ) . upper ( ) NEW_LINE str1 += c NEW_LINE DEDENT else : NEW_LINE INDENT str1 += str [ i ] NEW_LINE DEDENT DEDENT print ( str1 ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT str = " eutopia " NEW_LINE conVowUpp ( str ) NEW_LINE DEDENT
mp = { } NEW_LINE N , P = 0 , 0 NEW_LINE
def helper ( mid ) : NEW_LINE INDENT cnt = 0 ; NEW_LINE for i in mp : NEW_LINE INDENT temp = mp [ i ] NEW_LINE while ( temp >= mid ) : NEW_LINE INDENT temp -= mid NEW_LINE cnt += 1 NEW_LINE DEDENT DEDENT DEDENT
return cnt >= N NEW_LINE
def findMaximumDays ( arr ) : NEW_LINE
for i in range ( P ) : NEW_LINE INDENT mp [ arr [ i ] ] = mp . get ( arr [ i ] , 0 ) + 1 NEW_LINE DEDENT
start = 0 NEW_LINE end = P NEW_LINE ans = 0 NEW_LINE while ( start <= end ) : NEW_LINE
mid = start + ( ( end - start ) // 2 ) NEW_LINE
if ( mid != 0 and helper ( mid ) ) : NEW_LINE INDENT ans = mid NEW_LINE DEDENT
start = mid + 1 NEW_LINE elif ( mid == 0 ) : NEW_LINE start = mid + 1 NEW_LINE else : NEW_LINE end = mid - 1 NEW_LINE return ans NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 3 NEW_LINE P = 10 NEW_LINE arr = [ 1 , 2 , 2 , 1 , 1 , 3 , 3 , 3 , 2 , 4 ] NEW_LINE DEDENT
print ( findMaximumDays ( arr ) ) NEW_LINE
def countSubarrays ( a , n , k ) : NEW_LINE
ans = 0 NEW_LINE
pref = [ ] NEW_LINE pref . append ( 0 ) NEW_LINE
for i in range ( n ) : NEW_LINE INDENT pref . append ( ( a [ i ] + pref [ i ] ) % k ) NEW_LINE DEDENT
for i in range ( 1 , n + 1 , 1 ) : NEW_LINE INDENT for j in range ( i , n + 1 , 1 ) : NEW_LINE DEDENT
if ( ( pref [ j ] - pref [ i - 1 ] + k ) % k == j - i + 1 ) : NEW_LINE INDENT ans += 1 NEW_LINE DEDENT
print ( ans , end = ' ▁ ' ) NEW_LINE
arr = [ 2 , 3 , 5 , 3 , 1 , 5 ] NEW_LINE
N = len ( arr ) NEW_LINE
K = 4 NEW_LINE
countSubarrays ( arr , N , K ) NEW_LINE
def countSubarrays ( a , n , k ) : NEW_LINE
cnt = { } NEW_LINE
ans = 0 NEW_LINE
pref = [ ] NEW_LINE pref . append ( 0 ) NEW_LINE
for i in range ( n ) : NEW_LINE INDENT pref . append ( ( a [ i ] + pref [ i ] ) % k ) NEW_LINE DEDENT
cnt [ 0 ] = 1 NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE
remIdx = i - k NEW_LINE if ( remIdx >= 0 ) : NEW_LINE INDENT if ( ( pref [ remIdx ] - remIdx % k + k ) % k in cnt ) : NEW_LINE INDENT cnt [ ( pref [ remIdx ] - remIdx % k + k ) % k ] -= 1 NEW_LINE DEDENT else : NEW_LINE INDENT cnt [ ( pref [ remIdx ] - remIdx % k + k ) % k ] = - 1 NEW_LINE DEDENT DEDENT
if ( pref [ i ] - i % k + k ) % k in cnt : NEW_LINE INDENT ans += cnt [ ( pref [ i ] - i % k + k ) % k ] NEW_LINE DEDENT
if ( pref [ i ] - i % k + k ) % k in cnt : NEW_LINE INDENT cnt [ ( pref [ i ] - i % k + k ) % k ] += 1 NEW_LINE DEDENT else : NEW_LINE INDENT cnt [ ( pref [ i ] - i % k + k ) % k ] = 1 NEW_LINE DEDENT
print ( ans , end = ' ▁ ' ) NEW_LINE
arr = [ 2 , 3 , 5 , 3 , 1 , 5 ] NEW_LINE
N = len ( arr ) NEW_LINE
K = 4 NEW_LINE
countSubarrays ( arr , N , K ) NEW_LINE
def check ( s , k ) : NEW_LINE INDENT n = len ( s ) NEW_LINE DEDENT
for i in range ( k ) : NEW_LINE INDENT for j in range ( i , n , k ) : NEW_LINE DEDENT
if ( s [ i ] != s [ j ] ) : NEW_LINE INDENT return False NEW_LINE DEDENT c = 0 NEW_LINE
for i in range ( k ) : NEW_LINE
if ( s [ i ] == '0' ) : NEW_LINE
c += 1 NEW_LINE
else : NEW_LINE
c -= 1 NEW_LINE
if ( c == 0 ) : NEW_LINE INDENT return True NEW_LINE DEDENT else : NEW_LINE INDENT return False NEW_LINE DEDENT
s = "101010" NEW_LINE k = 2 NEW_LINE if ( check ( s , k ) != 0 ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
from collections import defaultdict NEW_LINE
def isSame ( str , n ) : NEW_LINE
mp = defaultdict ( lambda : 0 ) NEW_LINE for i in range ( len ( str ) ) : NEW_LINE INDENT mp [ ord ( str [ i ] ) - ord ( ' a ' ) ] += 1 NEW_LINE DEDENT for it in mp . keys ( ) : NEW_LINE
if ( mp [ it ] >= n ) : NEW_LINE INDENT return True NEW_LINE DEDENT
return False NEW_LINE
str = " ccabcba " NEW_LINE n = 4 NEW_LINE
if ( isSame ( str , n ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
import math NEW_LINE eps = 1e-6 NEW_LINE
def func ( a , b , c , x ) : NEW_LINE INDENT return a * x * x + b * x + c NEW_LINE DEDENT
def findRoot ( a , b , c , low , high ) : NEW_LINE INDENT x = - 1 NEW_LINE DEDENT
while abs ( high - low ) > eps : NEW_LINE
x = ( low + high ) / 2 NEW_LINE
if ( func ( a , b , c , low ) * func ( a , b , c , x ) <= 0 ) : NEW_LINE INDENT high = x NEW_LINE DEDENT
else : NEW_LINE INDENT low = x NEW_LINE DEDENT
return x NEW_LINE
def solve ( a , b , c , A , B ) : NEW_LINE
if ( func ( a , b , c , A ) * func ( a , b , c , B ) > 0 ) : NEW_LINE INDENT print ( " No ▁ solution " ) NEW_LINE DEDENT
else : NEW_LINE INDENT print ( " { : . 4f } " . format ( findRoot ( a , b , c , A , B ) ) ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE
a = 2 NEW_LINE b = - 3 NEW_LINE c = - 2 NEW_LINE A = 0 NEW_LINE B = 3 NEW_LINE
solve ( a , b , c , A , B ) NEW_LINE
def possible ( mid , a ) : NEW_LINE
n = len ( a ) ; NEW_LINE
total = ( n * ( n - 1 ) ) // 2 ; NEW_LINE
need = ( total + 1 ) // 2 ; NEW_LINE count = 0 ; NEW_LINE start = 0 ; end = 1 ; NEW_LINE
while ( end < n ) : NEW_LINE INDENT if ( a [ end ] - a [ start ] <= mid ) : NEW_LINE INDENT end += 1 ; NEW_LINE DEDENT else : NEW_LINE INDENT count += ( end - start - 1 ) ; NEW_LINE start += 1 ; NEW_LINE DEDENT DEDENT
if ( end == n and start < end and a [ end - 1 ] - a [ start ] <= mid ) : NEW_LINE INDENT t = end - start - 1 ; NEW_LINE count += ( t * ( t + 1 ) // 2 ) ; NEW_LINE DEDENT
if ( count >= need ) : NEW_LINE INDENT return True ; NEW_LINE DEDENT else : NEW_LINE INDENT return False ; NEW_LINE DEDENT
def findMedian ( a ) : NEW_LINE
n = len ( a ) ; NEW_LINE
low = 0 ; high = a [ n - 1 ] - a [ 0 ] ; NEW_LINE
while ( low <= high ) : NEW_LINE
mid = ( low + high ) // 2 ; NEW_LINE
if ( possible ( mid , a ) ) : NEW_LINE INDENT high = mid - 1 ; NEW_LINE DEDENT else : NEW_LINE INDENT low = mid + 1 ; NEW_LINE DEDENT
return high + 1 ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = [ 1 , 7 , 5 , 2 ] ; NEW_LINE a . sort ( ) NEW_LINE print ( findMedian ( a ) ) ; NEW_LINE DEDENT
def UniversalSubset ( A , B ) : NEW_LINE
n1 = len ( A ) NEW_LINE n2 = len ( B ) NEW_LINE
res = [ ] NEW_LINE
A_freq = [ [ 0 for x in range ( 26 ) ] for y in range ( n1 ) ] NEW_LINE
for i in range ( n1 ) : NEW_LINE INDENT for j in range ( len ( A [ i ] ) ) : NEW_LINE INDENT A_freq [ i ] [ ord ( A [ i ] [ j ] ) - ord ( ' a ' ) ] += 1 NEW_LINE DEDENT DEDENT
B_freq = [ 0 ] * 26 NEW_LINE for i in range ( n2 ) : NEW_LINE INDENT arr = [ 0 ] * 26 NEW_LINE for j in range ( len ( B [ i ] ) ) : NEW_LINE INDENT arr [ ord ( B [ i ] [ j ] ) - ord ( ' a ' ) ] += 1 NEW_LINE B_freq [ ord ( B [ i ] [ j ] ) - ord ( ' a ' ) ] = max ( B_freq [ ord ( B [ i ] [ j ] ) - ord ( ' a ' ) ] , arr [ ord ( B [ i ] [ j ] ) - ord ( ' a ' ) ] ) NEW_LINE DEDENT DEDENT for i in range ( n1 ) : NEW_LINE INDENT flag = 0 NEW_LINE for j in range ( 26 ) : NEW_LINE DEDENT
if ( A_freq [ i ] [ j ] < B_freq [ j ] ) : NEW_LINE
flag = 1 NEW_LINE break NEW_LINE
if ( flag == 0 ) : NEW_LINE
res . append ( A [ i ] ) NEW_LINE
if ( len ( res ) ) : NEW_LINE
for i in range ( len ( res ) ) : NEW_LINE INDENT for j in range ( len ( res [ i ] ) ) : NEW_LINE INDENT print ( res [ i ] [ j ] , end = " " ) NEW_LINE DEDENT DEDENT
else : NEW_LINE INDENT print ( - 1 , end = " " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT A = [ " geeksforgeeks " , " topcoder " , " leetcode " ] NEW_LINE B = [ " geek " , " ee " ] NEW_LINE UniversalSubset ( A , B ) NEW_LINE DEDENT
import sys NEW_LINE
def findPair ( a , n ) : NEW_LINE
min_dist = sys . maxsize NEW_LINE index_a = - 1 NEW_LINE index_b = - 1 NEW_LINE
for i in range ( n ) : NEW_LINE
for j in range ( i + 1 , n ) : NEW_LINE
if ( j - i < min_dist ) : NEW_LINE
if ( ( a [ i ] % a [ j ] == 0 ) or ( a [ j ] % a [ i ] == 0 ) ) : NEW_LINE
min_dist = j - i NEW_LINE
index_a = i NEW_LINE index_b = j NEW_LINE
if ( index_a == - 1 ) : NEW_LINE INDENT print ( " - 1" ) NEW_LINE DEDENT
else : NEW_LINE INDENT print ( " ( " , a [ index_a ] , " , ▁ " , a [ index_b ] , " ) " ) NEW_LINE DEDENT
a = [ 2 , 3 , 4 , 5 , 6 ] NEW_LINE n = len ( a ) NEW_LINE
findPair ( a , n ) NEW_LINE
def printNum ( L , R ) : NEW_LINE
for i in range ( L , R + 1 ) : NEW_LINE INDENT temp = i NEW_LINE c = 10 NEW_LINE flag = 0 NEW_LINE DEDENT
while ( temp > 0 ) : NEW_LINE
if ( temp % 10 >= c ) : NEW_LINE INDENT flag = 1 NEW_LINE break NEW_LINE DEDENT c = temp % 10 NEW_LINE temp //= 10 NEW_LINE
if ( flag == 0 ) : NEW_LINE INDENT print ( i , end = " ▁ " ) NEW_LINE DEDENT
L = 10 NEW_LINE R = 15 NEW_LINE
printNum ( L , R ) NEW_LINE
import sys NEW_LINE
def findMissing ( arr , left , right , diff ) : NEW_LINE
if ( right <= left ) : NEW_LINE INDENT return sys . maxsize NEW_LINE DEDENT
mid = left + ( right - left ) // 2 NEW_LINE
if ( arr [ mid + 1 ] - arr [ mid ] != diff ) : NEW_LINE INDENT return ( arr [ mid ] + diff ) NEW_LINE DEDENT
if ( mid > 0 and arr [ mid ] - arr [ mid - 1 ] != diff ) : NEW_LINE INDENT return ( arr [ mid - 1 ] + diff ) NEW_LINE DEDENT
if ( arr [ mid ] == arr [ 0 ] + mid * diff ) : NEW_LINE INDENT return findMissing ( arr , mid + 1 , right , diff ) NEW_LINE DEDENT
return findMissing ( arr , left , mid - 1 , diff ) NEW_LINE
def missingElement ( arr , n ) : NEW_LINE
arr . sort ( ) NEW_LINE
diff = ( arr [ n - 1 ] - arr [ 0 ] ) // n NEW_LINE
return findMissing ( arr , 0 , n - 1 , diff ) NEW_LINE
arr = [ 2 , 8 , 6 , 10 ] NEW_LINE n = len ( arr ) NEW_LINE
print ( missingElement ( arr , n ) ) NEW_LINE
def power ( x , y ) : NEW_LINE INDENT if ( y == 0 ) : NEW_LINE INDENT return 1 ; NEW_LINE DEDENT temp = power ( x , y // 2 ) ; NEW_LINE if ( y % 2 == 0 ) : NEW_LINE INDENT return temp * temp ; NEW_LINE DEDENT else : NEW_LINE INDENT return x * temp * temp ; NEW_LINE DEDENT DEDENT
def nthRootSearch ( low , high , N , K ) : NEW_LINE
if ( low <= high ) : NEW_LINE
mid = ( low + high ) // 2 ; NEW_LINE
if ( ( power ( mid , K ) <= N ) and ( power ( mid + 1 , K ) > N ) ) : NEW_LINE INDENT return mid ; NEW_LINE DEDENT
elif ( power ( mid , K ) < N ) : NEW_LINE INDENT return nthRootSearch ( mid + 1 , high , N , K ) ; NEW_LINE DEDENT else : NEW_LINE INDENT return nthRootSearch ( low , mid - 1 , N , K ) ; NEW_LINE DEDENT return low ; NEW_LINE
N = 16 ; K = 4 ; NEW_LINE
print ( nthRootSearch ( 0 , N , N , K ) ) NEW_LINE
def get_subset_count ( arr , K , N ) : NEW_LINE
arr . sort ( ) NEW_LINE left = 0 ; NEW_LINE right = N - 1 ; NEW_LINE
ans = 0 ; NEW_LINE while ( left <= right ) : NEW_LINE INDENT if ( arr [ left ] + arr [ right ] < K ) : NEW_LINE DEDENT
ans += 1 << ( right - left ) ; NEW_LINE left += 1 ; NEW_LINE else : NEW_LINE
right -= 1 ; NEW_LINE return ans ; NEW_LINE
arr = [ 2 , 4 , 5 , 7 ] ; NEW_LINE K = 8 ; NEW_LINE print ( get_subset_count ( arr , K , 4 ) ) NEW_LINE
def minMaxDiff ( arr , n , k ) : NEW_LINE INDENT max_adj_dif = float ( ' - inf ' ) ; NEW_LINE DEDENT
for i in range ( n - 1 ) : NEW_LINE INDENT max_adj_dif = max ( max_adj_dif , abs ( arr [ i ] - arr [ i + 1 ] ) ) ; NEW_LINE DEDENT
if ( max_adj_dif == 0 ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT
best = 1 ; NEW_LINE worst = max_adj_dif ; NEW_LINE while ( best < worst ) : NEW_LINE INDENT mid = ( best + worst ) // 2 ; NEW_LINE DEDENT
required = 0 NEW_LINE for i in range ( n - 1 ) : NEW_LINE INDENT required += ( abs ( arr [ i ] - arr [ i + 1 ] ) - 1 ) // mid NEW_LINE DEDENT
if ( required > k ) : NEW_LINE INDENT best = mid + 1 ; NEW_LINE DEDENT
else : NEW_LINE INDENT worst = mid NEW_LINE DEDENT return worst NEW_LINE
arr = [ 3 , 12 , 25 , 50 ] NEW_LINE n = len ( arr ) NEW_LINE k = 7 NEW_LINE print ( minMaxDiff ( arr , n , k ) ) NEW_LINE
import math NEW_LINE
def checkMin ( arr , n ) : NEW_LINE
smallest = math . inf NEW_LINE secondSmallest = math . inf NEW_LINE for i in range ( n ) : NEW_LINE
if ( arr [ i ] < smallest ) : NEW_LINE INDENT secondSmallest = smallest NEW_LINE smallest = arr [ i ] NEW_LINE DEDENT
elif ( arr [ i ] < secondSmallest ) : NEW_LINE INDENT secondSmallest = arr [ i ] NEW_LINE DEDENT if ( 2 * smallest <= secondSmallest ) : NEW_LINE print ( " Yes " ) NEW_LINE else : NEW_LINE print ( " No " ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 2 , 3 , 4 , 5 ] NEW_LINE n = len ( arr ) NEW_LINE checkMin ( arr , n ) NEW_LINE DEDENT
import sys NEW_LINE
def createHash ( hash , maxElement ) : NEW_LINE
prev = 0 NEW_LINE curr = 1 NEW_LINE hash . add ( prev ) NEW_LINE hash . add ( curr ) NEW_LINE while ( curr <= maxElement ) : NEW_LINE
temp = curr + prev NEW_LINE hash . add ( temp ) NEW_LINE
prev = curr NEW_LINE curr = temp NEW_LINE
def fibonacci ( arr , n ) : NEW_LINE
max_val = max ( arr ) NEW_LINE
hash = set ( ) NEW_LINE createHash ( hash , max_val ) NEW_LINE
minimum = sys . maxsize NEW_LINE maximum = - sys . maxsize - 1 NEW_LINE for i in range ( n ) : NEW_LINE
if ( arr [ i ] in hash ) : NEW_LINE
minimum = min ( minimum , arr [ i ] ) NEW_LINE maximum = max ( maximum , arr [ i ] ) NEW_LINE print ( minimum , end = " , ▁ " ) NEW_LINE print ( maximum ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 1 , 2 , 3 , 4 , 5 , 6 , 7 ] NEW_LINE n = len ( arr ) NEW_LINE fibonacci ( arr , n ) NEW_LINE DEDENT
def isValidLen ( s , lenn , k ) : NEW_LINE
n = len ( s ) NEW_LINE
mp = dict ( ) NEW_LINE right = 0 NEW_LINE
while ( right < lenn ) : NEW_LINE INDENT mp [ s [ right ] ] = mp . get ( s [ right ] , 0 ) + 1 NEW_LINE right += 1 NEW_LINE DEDENT if ( len ( mp ) <= k ) : NEW_LINE INDENT return True NEW_LINE DEDENT
while ( right < n ) : NEW_LINE
mp [ s [ right ] ] = mp . get ( s [ right ] , 0 ) + 1 NEW_LINE
mp [ s [ right - lenn ] ] -= 1 NEW_LINE
if ( mp [ s [ right - lenn ] ] == 0 ) : NEW_LINE INDENT del mp [ s [ right - lenn ] ] NEW_LINE DEDENT if ( len ( mp ) <= k ) : NEW_LINE INDENT return True NEW_LINE DEDENT right += 1 NEW_LINE return len ( mp ) <= k NEW_LINE
def maxLenSubStr ( s , k ) : NEW_LINE
uni = dict ( ) NEW_LINE for x in s : NEW_LINE INDENT uni [ x ] = 1 NEW_LINE DEDENT if ( len ( uni ) < k ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
n = len ( s ) NEW_LINE
lo = - 1 NEW_LINE hi = n + 1 NEW_LINE while ( hi - lo > 1 ) : NEW_LINE INDENT mid = lo + hi >> 1 NEW_LINE if ( isValidLen ( s , mid , k ) ) : NEW_LINE INDENT lo = mid NEW_LINE DEDENT else : NEW_LINE INDENT hi = mid NEW_LINE DEDENT DEDENT return lo NEW_LINE
s = " aabacbebebe " NEW_LINE k = 3 NEW_LINE print ( maxLenSubStr ( s , k ) ) NEW_LINE
def isSquarePossible ( arr , n , l ) : NEW_LINE
cnt = 0 NEW_LINE for i in range ( n ) : NEW_LINE
if arr [ i ] >= l : NEW_LINE INDENT cnt += 1 NEW_LINE DEDENT
if cnt >= l : NEW_LINE INDENT return True NEW_LINE DEDENT return False NEW_LINE
def maxArea ( arr , n ) : NEW_LINE INDENT l , r = 0 , n NEW_LINE len = 0 NEW_LINE while l <= r : NEW_LINE INDENT m = l + ( ( r - l ) // 2 ) NEW_LINE DEDENT DEDENT
if isSquarePossible ( arr , n , m ) : NEW_LINE INDENT len = m NEW_LINE l = m + 1 NEW_LINE DEDENT
else : NEW_LINE INDENT r = m - 1 NEW_LINE DEDENT
return ( len * len ) NEW_LINE
arr = [ 1 , 3 , 4 , 5 , 5 ] NEW_LINE n = len ( arr ) NEW_LINE print ( maxArea ( arr , n ) ) NEW_LINE
def insertNames ( arr , n ) : NEW_LINE
string = set ( ) ; NEW_LINE for i in range ( n ) : NEW_LINE
if arr [ i ] not in string : NEW_LINE INDENT print ( " No " ) ; NEW_LINE string . add ( arr [ i ] ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Yes " ) ; NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ " geeks " , " for " , " geeks " ] ; NEW_LINE n = len ( arr ) ; NEW_LINE insertNames ( arr , n ) ; NEW_LINE DEDENT
def countLessThan ( arr , n , key ) : NEW_LINE INDENT l = 0 NEW_LINE r = n - 1 NEW_LINE index = - 1 NEW_LINE DEDENT
while ( l <= r ) : NEW_LINE INDENT m = ( l + r ) // 2 NEW_LINE if ( arr [ m ] < key ) : NEW_LINE INDENT l = m + 1 NEW_LINE index = m NEW_LINE DEDENT else : NEW_LINE INDENT r = m - 1 NEW_LINE DEDENT DEDENT return ( index + 1 ) NEW_LINE
def countGreaterThan ( arr , n , key ) : NEW_LINE INDENT l = 0 NEW_LINE r = n - 1 NEW_LINE index = - 1 NEW_LINE DEDENT
while ( l <= r ) : NEW_LINE INDENT m = ( l + r ) // 2 NEW_LINE if ( arr [ m ] <= key ) : NEW_LINE INDENT l = m + 1 NEW_LINE DEDENT else : NEW_LINE INDENT r = m - 1 NEW_LINE index = m NEW_LINE DEDENT DEDENT if ( index == - 1 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT return ( n - index ) NEW_LINE
def countTriplets ( n , a , b , c ) : NEW_LINE
a . sort NEW_LINE b . sort ( ) NEW_LINE c . sort ( ) NEW_LINE count = 0 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT current = b [ i ] NEW_LINE a_index = - 1 NEW_LINE c_index = - 1 NEW_LINE DEDENT
low = countLessThan ( a , n , current ) NEW_LINE
high = countGreaterThan ( c , n , current ) NEW_LINE
count += ( low * high ) NEW_LINE return count NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = [ 1 , 5 ] NEW_LINE b = [ 2 , 4 ] NEW_LINE c = [ 3 , 6 ] NEW_LINE size = len ( a ) NEW_LINE print ( countTriplets ( size , a , b , c ) ) NEW_LINE DEDENT
def costToBalance ( s ) : NEW_LINE INDENT if ( len ( s ) == 0 ) : NEW_LINE INDENT print ( 0 ) NEW_LINE DEDENT DEDENT
ans = 0 NEW_LINE
' NEW_LINE INDENT o = 0 NEW_LINE c = 0 NEW_LINE for i in range ( len ( s ) ) : NEW_LINE INDENT if ( s [ i ] == ' ( ' ) : NEW_LINE INDENT o += 1 NEW_LINE DEDENT if ( s [ i ] == ' ) ' ) : NEW_LINE INDENT c += 1 NEW_LINE DEDENT DEDENT if ( o != c ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT a = [ 0 for i in range ( len ( s ) ) ] NEW_LINE if ( s [ 0 ] == ' ( ' ) : NEW_LINE INDENT a [ 0 ] = 1 NEW_LINE DEDENT else : NEW_LINE INDENT a [ 0 ] = - 1 NEW_LINE DEDENT if ( a [ 0 ] < 0 ) : NEW_LINE INDENT ans += abs ( a [ 0 ] ) NEW_LINE DEDENT for i in range ( 1 , len ( s ) ) : NEW_LINE INDENT if ( s [ i ] == ' ( ' ) : NEW_LINE INDENT a [ i ] = a [ i - 1 ] + 1 NEW_LINE DEDENT else : NEW_LINE INDENT a [ i ] = a [ i - 1 ] - 1 NEW_LINE DEDENT if ( a [ i ] < 0 ) : NEW_LINE INDENT ans += abs ( a [ i ] ) NEW_LINE DEDENT DEDENT return ans NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : s = " ) ) ) ( ( ( " NEW_LINE INDENT print ( costToBalance ( s ) ) s = " ) ) ( ( " NEW_LINE print ( costToBalance ( s ) ) NEW_LINE DEDENT
def middleOfThree ( a , b , c ) : NEW_LINE
x = a - b NEW_LINE
y = b - c NEW_LINE
z = a - c NEW_LINE
if x * y > 0 : NEW_LINE INDENT return b NEW_LINE DEDENT
elif ( x * z > 0 ) : NEW_LINE INDENT return NEW_LINE DEDENT else : NEW_LINE INDENT return a NEW_LINE DEDENT
a = 20 NEW_LINE b = 30 NEW_LINE c = 40 NEW_LINE print ( middleOfThree ( a , b , c ) ) NEW_LINE
def missing4 ( arr ) : NEW_LINE
helper = [ 0 ] * 4 NEW_LINE
for i in range ( 0 , len ( arr ) ) : NEW_LINE INDENT temp = abs ( arr [ i ] ) NEW_LINE DEDENT
if ( temp <= len ( arr ) ) : NEW_LINE INDENT arr [ temp - 1 ] = arr [ temp - 1 ] * ( - 1 ) NEW_LINE DEDENT
elif ( temp > len ( arr ) ) : NEW_LINE INDENT if ( temp % len ( arr ) ) : NEW_LINE INDENT helper [ temp % len ( arr ) - 1 ] = - 1 NEW_LINE DEDENT else : NEW_LINE INDENT helper [ ( temp % len ( arr ) ) + len ( arr ) - 1 ] = - 1 NEW_LINE DEDENT DEDENT
for i in range ( 0 , len ( arr ) ) : NEW_LINE INDENT if ( arr [ i ] > 0 ) : NEW_LINE INDENT print ( ( i + 1 ) , end = " ▁ " ) NEW_LINE DEDENT DEDENT for i in range ( 0 , len ( helper ) ) : NEW_LINE INDENT if ( helper [ i ] >= 0 ) : NEW_LINE INDENT print ( ( len ( arr ) + i + 1 ) , end = " ▁ " ) NEW_LINE DEDENT DEDENT
arr = [ 1 , 7 , 3 , 12 , 5 , 10 , 8 , 4 , 9 ] NEW_LINE missing4 ( arr ) NEW_LINE
def lexiMiddleSmallest ( K , N ) : NEW_LINE
if ( K % 2 == 0 ) : NEW_LINE
print ( K // 2 , end = " ▁ " ) NEW_LINE
for i in range ( N - 1 ) : NEW_LINE INDENT print ( K , end = " ▁ " ) NEW_LINE DEDENT print ( ) NEW_LINE return NEW_LINE
a = [ ( K + 1 ) // 2 ] * ( N ) NEW_LINE
for i in range ( N // 2 ) : NEW_LINE
if ( a [ - 1 ] == 1 ) : NEW_LINE
del a [ - 1 ] NEW_LINE
else : NEW_LINE
a [ - 1 ] -= 1 NEW_LINE
while ( len ( a ) < N ) : NEW_LINE INDENT a . append ( K ) NEW_LINE DEDENT
for i in a : NEW_LINE INDENT print ( i , end = " ▁ " ) NEW_LINE DEDENT print ( ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT K , N = 2 , 4 NEW_LINE lexiMiddleSmallest ( K , N ) NEW_LINE DEDENT
def findLastElement ( arr , N ) : NEW_LINE
arr . sort ( ) ; NEW_LINE i = 0 ; NEW_LINE
for i in range ( 1 , N ) : NEW_LINE
if ( arr [ i ] - arr [ i - 1 ] != 0 \ and arr [ i ] - arr [ i - 1 ] != 2 ) : NEW_LINE INDENT print ( " - 1" ) ; NEW_LINE return ; NEW_LINE DEDENT
print ( arr [ N - 1 ] ) ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 2 , 4 , 6 , 8 , 0 , 8 ] ; NEW_LINE N = len ( arr ) ; NEW_LINE findLastElement ( arr , N ) ; NEW_LINE DEDENT
def maxDivisions ( arr , N , X ) : NEW_LINE
arr . sort ( reverse = True ) NEW_LINE
maxSub = 0 ; NEW_LINE
size = 0 ; NEW_LINE
for i in range ( N ) : NEW_LINE
size += 1 ; NEW_LINE
if ( arr [ i ] * size >= X ) : NEW_LINE
maxSub += 1 ; NEW_LINE
size = 0 ; NEW_LINE print ( maxSub ) ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
arr = [ 1 , 3 , 3 , 7 ] ; NEW_LINE
N = len ( arr ) ; NEW_LINE
X = 3 ; NEW_LINE maxDivisions ( arr , N , X ) ; NEW_LINE
def maxPossibleSum ( arr , N ) : NEW_LINE
arr . sort ( ) NEW_LINE sum = 0 NEW_LINE j = N - 3 NEW_LINE while ( j >= 0 ) : NEW_LINE
sum += arr [ j ] NEW_LINE j -= 3 NEW_LINE
print ( sum ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
arr = [ 7 , 4 , 5 , 2 , 3 , 1 , 5 , 9 ] NEW_LINE
N = 8 NEW_LINE maxPossibleSum ( arr , N ) NEW_LINE
def insertionSort ( arr , n ) : NEW_LINE INDENT i = 0 NEW_LINE key = 0 NEW_LINE j = 0 NEW_LINE for i in range ( 1 , n , 1 ) : NEW_LINE INDENT key = arr [ i ] NEW_LINE j = i - 1 NEW_LINE DEDENT DEDENT
while ( j >= 0 and arr [ j ] > key ) : NEW_LINE INDENT arr [ j + 1 ] = arr [ j ] NEW_LINE j = j - 1 NEW_LINE DEDENT arr [ j + 1 ] = key NEW_LINE
def printArray ( arr , n ) : NEW_LINE INDENT i = 0 NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT print ( " " , end ▁ = ▁ " " ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 12 , 11 , 13 , 5 , 6 ] NEW_LINE N = len ( arr ) NEW_LINE DEDENT
insertionSort ( arr , N ) NEW_LINE printArray ( arr , N ) NEW_LINE
def getPairs ( arr , N , K ) : NEW_LINE
count = 0 NEW_LINE
for i in range ( N ) : NEW_LINE INDENT for j in range ( i + 1 , N ) : NEW_LINE DEDENT
if ( arr [ i ] > K * arr [ i + 1 ] ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT print ( count ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 5 , 6 , 2 , 1 ] NEW_LINE N = len ( arr ) NEW_LINE K = 2 NEW_LINE DEDENT
getPairs ( arr , N , K ) NEW_LINE
def merge ( arr , temp , l , m , r , K ) : NEW_LINE
i = l NEW_LINE
j = m + 1 NEW_LINE
cnt = 0 NEW_LINE for l in range ( m + 1 ) : NEW_LINE INDENT found = False NEW_LINE DEDENT
while ( j <= r ) : NEW_LINE
if ( arr [ i ] >= K * arr [ j ] ) : NEW_LINE INDENT found = True NEW_LINE DEDENT else : NEW_LINE INDENT break NEW_LINE DEDENT j += 1 NEW_LINE
if ( found ) : NEW_LINE INDENT cnt += j - ( m + 1 ) NEW_LINE j -= 1 NEW_LINE DEDENT
k = l NEW_LINE i = l NEW_LINE j = m + 1 NEW_LINE while ( i <= m and j <= r ) : NEW_LINE INDENT if ( arr [ i ] <= arr [ j ] ) : NEW_LINE INDENT temp [ k ] = arr [ i ] NEW_LINE k += 1 NEW_LINE i += 1 NEW_LINE DEDENT else : NEW_LINE INDENT temp [ k ] = arr [ j ] NEW_LINE k += 1 NEW_LINE j += 1 NEW_LINE DEDENT DEDENT
while ( i <= m ) : NEW_LINE INDENT temp [ k ] = arr [ i ] NEW_LINE k += 1 NEW_LINE i += 1 NEW_LINE DEDENT
while ( j <= r ) : NEW_LINE INDENT temp [ k ] = arr [ j ] NEW_LINE k += 1 NEW_LINE j += 1 NEW_LINE DEDENT for i in range ( l , r + 1 ) : NEW_LINE INDENT arr [ i ] = temp [ i ] NEW_LINE DEDENT
return cnt NEW_LINE
def mergeSortUtil ( arr , temp , l , r , K ) : NEW_LINE INDENT cnt = 0 NEW_LINE if ( l < r ) : NEW_LINE DEDENT
m = ( l + r ) // 2 NEW_LINE
cnt += mergeSortUtil ( arr , temp , l , m , K ) NEW_LINE cnt += mergeSortUtil ( arr , temp , m + 1 , r , K ) NEW_LINE
cnt += merge ( arr , temp , l , m , r , K ) NEW_LINE return cnt NEW_LINE
def mergeSort ( arr , N , K ) : NEW_LINE INDENT temp = [ 0 ] * N NEW_LINE print ( mergeSortUtil ( arr , temp , 0 , N - 1 , K ) ) NEW_LINE DEDENT
arr = [ 5 , 6 , 2 , 5 ] NEW_LINE N = len ( arr ) NEW_LINE K = 2 NEW_LINE
mergeSort ( arr , N , K ) NEW_LINE
def minRemovals ( A , N ) : NEW_LINE
A . sort ( ) NEW_LINE
mx = A [ N - 1 ] NEW_LINE
sum = 1 NEW_LINE
for i in range ( 0 , N ) : NEW_LINE INDENT sum += A [ i ] NEW_LINE DEDENT if ( ( sum - mx ) >= mx ) : NEW_LINE INDENT print ( 0 , end = " " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( 2 * mx - sum , end = " " ) NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT A = [ 3 , 3 , 2 ] NEW_LINE N = len ( A ) NEW_LINE DEDENT
minRemovals ( A , N ) NEW_LINE
def rearrangeArray ( a , n ) : NEW_LINE
a = sorted ( a ) NEW_LINE
for i in range ( n - 1 ) : NEW_LINE
if ( a [ i ] == i + 1 ) : NEW_LINE
a [ i ] , a [ i + 1 ] = a [ i + 1 ] , a [ i ] NEW_LINE
if ( a [ n - 1 ] == n ) : NEW_LINE
a [ n - 1 ] , a [ n - 2 ] = a [ n - 2 ] , a [ n - 1 ] NEW_LINE
for i in range ( n ) : NEW_LINE INDENT print ( a [ i ] , end = " ▁ " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 1 , 5 , 3 , 2 , 4 ] NEW_LINE N = len ( arr ) NEW_LINE DEDENT
rearrangeArray ( arr , N ) NEW_LINE
def minOperations ( arr1 , arr2 , i , j ) : NEW_LINE
if arr1 == arr2 : NEW_LINE INDENT return 0 NEW_LINE DEDENT if i >= len ( arr1 ) or j >= len ( arr2 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
if arr1 [ i ] < arr2 [ j ] : NEW_LINE
return 1 + minOperations ( arr1 , arr2 , i + 1 , j + 1 ) NEW_LINE
return max ( minOperations ( arr1 , arr2 , i , j + 1 ) , minOperations ( arr1 , arr2 , i + 1 , j ) ) NEW_LINE
def minOperationsUtil ( arr ) : NEW_LINE INDENT brr = sorted ( arr ) ; NEW_LINE DEDENT
if ( arr == brr ) : NEW_LINE
print ( "0" ) NEW_LINE
else : NEW_LINE
print ( minOperations ( arr , brr , 0 , 0 ) ) NEW_LINE
arr = [ 4 , 7 , 2 , 3 , 9 ] NEW_LINE minOperationsUtil ( arr ) NEW_LINE
def canTransform ( s , t ) : NEW_LINE INDENT n = len ( s ) NEW_LINE DEDENT
occur = [ [ ] for i in range ( 26 ) ] NEW_LINE for x in range ( n ) : NEW_LINE INDENT ch = ord ( s [ x ] ) - ord ( ' a ' ) NEW_LINE occur [ ch ] . append ( x ) NEW_LINE DEDENT
idx = [ 0 ] * ( 26 ) NEW_LINE poss = True NEW_LINE for x in range ( n ) : NEW_LINE INDENT ch = ord ( t [ x ] ) - ord ( ' a ' ) NEW_LINE DEDENT
if ( idx [ ch ] >= len ( occur [ ch ] ) ) : NEW_LINE
poss = False NEW_LINE break NEW_LINE for small in range ( ch ) : NEW_LINE
if ( idx [ small ] < len ( occur [ small ] ) and occur [ small ] [ idx [ small ] ] < occur [ ch ] [ idx [ ch ] ] ) : NEW_LINE
poss = False NEW_LINE break NEW_LINE idx [ ch ] += 1 NEW_LINE
if ( poss ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT s = " hdecb " NEW_LINE t = " cdheb " NEW_LINE canTransform ( s , t ) NEW_LINE DEDENT
def inversionCount ( s ) : NEW_LINE
freq = [ 0 for _ in range ( 26 ) ] NEW_LINE inv = 0 NEW_LINE for i in range ( len ( s ) ) : NEW_LINE
temp = 0 NEW_LINE for j in range ( ord ( s [ i ] ) - ord ( ' a ' ) ) : NEW_LINE
temp += freq [ j ] NEW_LINE inv += ( i - temp ) NEW_LINE
freq [ ord ( s [ i ] ) - ord ( ' a ' ) ] += 1 NEW_LINE return inv NEW_LINE
def haveRepeated ( S1 , S2 ) : NEW_LINE INDENT freq = [ 0 for _ in range ( 26 ) ] NEW_LINE for i in range ( len ( S1 ) ) : NEW_LINE INDENT if freq [ ord ( S1 [ i ] ) - ord ( ' a ' ) ] > 0 : NEW_LINE INDENT return 1 NEW_LINE DEDENT freq [ ord ( S1 [ i ] ) - ord ( ' a ' ) ] += 1 NEW_LINE DEDENT for i in range ( 26 ) : NEW_LINE INDENT freq [ i ] = 0 NEW_LINE DEDENT for i in range ( len ( S2 ) ) : NEW_LINE INDENT if freq [ ord ( S2 [ i ] ) - ord ( ' a ' ) ] > 0 : NEW_LINE INDENT return 1 NEW_LINE DEDENT freq [ ord ( S2 [ i ] ) - ord ( ' a ' ) ] += 1 NEW_LINE DEDENT return 0 NEW_LINE DEDENT
def checkToMakeEqual ( S1 , S2 ) : NEW_LINE
freq = [ 0 for _ in range ( 26 ) ] NEW_LINE for i in range ( len ( S1 ) ) : NEW_LINE
freq [ ord ( S1 [ i ] ) - ord ( ' a ' ) ] += 1 NEW_LINE flag = 0 NEW_LINE for i in range ( len ( S2 ) ) : NEW_LINE if freq [ ord ( S2 [ i ] ) - ord ( ' a ' ) ] == 0 : NEW_LINE
flag = 1 NEW_LINE break NEW_LINE
freq [ ord ( S2 [ i ] ) - ord ( ' a ' ) ] -= 1 NEW_LINE if flag == 1 : NEW_LINE
print ( " No " ) NEW_LINE return NEW_LINE
invCount1 = inversionCount ( S1 ) NEW_LINE invCount2 = inversionCount ( S2 ) NEW_LINE if ( ( invCount1 == invCount2 ) or ( ( invCount1 % 2 ) == ( invCount2 % 2 ) ) or haveRepeated ( S1 , S2 ) == 1 ) : NEW_LINE
print ( " Yes " ) NEW_LINE else : NEW_LINE print ( " No " ) NEW_LINE
S1 = " abbca " NEW_LINE S2 = " acabb " NEW_LINE checkToMakeEqual ( S1 , S2 ) NEW_LINE
import math NEW_LINE
def sortArr ( a , n ) : NEW_LINE
k = int ( math . log ( n , 2 ) ) NEW_LINE k = int ( pow ( 2 , k ) ) NEW_LINE
while ( k > 0 ) : NEW_LINE INDENT i = 0 NEW_LINE while i + k < n : NEW_LINE INDENT if a [ i ] > a [ i + k ] : NEW_LINE INDENT a [ i ] , a [ i + k ] = a [ i + k ] , a [ i ] NEW_LINE DEDENT i = i + 1 NEW_LINE DEDENT DEDENT
k = k // 2 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT print ( a [ i ] , end = " ▁ " ) NEW_LINE DEDENT
a = [ 5 , 20 , 30 , 40 , 36 , 33 , 25 , 15 , 10 ] NEW_LINE n = len ( a ) NEW_LINE
sortArr ( a , n ) NEW_LINE
def maximumSum ( arr , n , k ) : NEW_LINE
elt = n // k ; NEW_LINE sum = 0 ; NEW_LINE
arr . sort ( ) ; NEW_LINE count = 0 ; NEW_LINE i = n - 1 ; NEW_LINE
while ( count < k ) : NEW_LINE INDENT sum += arr [ i ] ; NEW_LINE i -= 1 ; NEW_LINE count += 1 ; NEW_LINE DEDENT count = 0 ; NEW_LINE i = 0 ; NEW_LINE
while ( count < k ) : NEW_LINE INDENT sum += arr [ i ] ; NEW_LINE i += elt - 1 ; NEW_LINE count += 1 ; NEW_LINE DEDENT
print ( sum ) ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT Arr = [ 1 , 13 , 7 , 17 , 6 , 5 ] ; NEW_LINE K = 2 ; NEW_LINE size = len ( Arr ) ; NEW_LINE maximumSum ( Arr , size , K ) ; NEW_LINE DEDENT
def findMinSum ( arr , K , L , size ) : NEW_LINE INDENT if ( K * L > size ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT minsum = 0 NEW_LINE DEDENT
arr . sort ( ) NEW_LINE
for i in range ( K ) : NEW_LINE INDENT minsum += arr [ i ] NEW_LINE DEDENT
return minsum NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 2 , 15 , 5 , 1 , 35 , 16 , 67 , 10 ] NEW_LINE K = 3 NEW_LINE L = 2 NEW_LINE length = len ( arr ) NEW_LINE print ( findMinSum ( arr , K , L , length ) ) NEW_LINE DEDENT
def findKthSmallest ( arr , n , k ) : NEW_LINE
max = 0 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT if ( arr [ i ] > max ) : NEW_LINE INDENT max = arr [ i ] NEW_LINE DEDENT DEDENT
counter = [ 0 ] * ( max + 1 ) NEW_LINE
smallest = 0 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT counter [ arr [ i ] ] += 1 NEW_LINE DEDENT
for num in range ( 1 , max + 1 ) : NEW_LINE
if ( counter [ num ] > 0 ) : NEW_LINE
smallest += counter [ num ] NEW_LINE
if ( smallest >= k ) : NEW_LINE
return num NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
arr = [ 7 , 1 , 4 , 4 , 20 , 15 , 8 ] NEW_LINE N = len ( arr ) NEW_LINE K = 5 NEW_LINE
print ( findKthSmallest ( arr , N , K ) ) NEW_LINE
def lexNumbers ( n ) : NEW_LINE INDENT s = [ ] NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE INDENT s . append ( str ( i ) ) NEW_LINE DEDENT s . sort ( ) NEW_LINE ans = [ ] NEW_LINE for i in range ( n ) : NEW_LINE INDENT ans . append ( int ( s [ i ] ) ) NEW_LINE DEDENT for i in range ( n ) : NEW_LINE INDENT print ( ans [ i ] , end = ' ▁ ' ) NEW_LINE DEDENT DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 15 NEW_LINE lexNumbers ( n ) NEW_LINE DEDENT
N = 4 NEW_LINE def func ( a ) : NEW_LINE
for i in range ( N ) : NEW_LINE
if i % 2 == 0 : NEW_LINE INDENT for j in range ( N ) : NEW_LINE INDENT for k in range ( j + 1 , N ) : NEW_LINE DEDENT DEDENT
if a [ i ] [ j ] > a [ i ] [ k ] : NEW_LINE
temp = a [ i ] [ j ] NEW_LINE a [ i ] [ j ] = a [ i ] [ k ] NEW_LINE a [ i ] [ k ] = temp NEW_LINE
else : NEW_LINE INDENT for j in range ( N ) : NEW_LINE INDENT for k in range ( j + 1 , N ) : NEW_LINE DEDENT DEDENT
if a [ i ] [ j ] < a [ i ] [ k ] : NEW_LINE
temp = a [ i ] [ j ] NEW_LINE a [ i ] [ j ] = a [ i ] [ k ] NEW_LINE a [ i ] [ k ] = temp NEW_LINE
for i in range ( N ) : NEW_LINE INDENT for j in range ( N ) : NEW_LINE INDENT print ( a [ i ] [ j ] , end = " ▁ " ) NEW_LINE DEDENT print ( ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT a = [ [ 5 , 7 , 3 , 4 ] , [ 9 , 5 , 8 , 2 ] , [ 6 , 3 , 8 , 1 ] , [ 5 , 8 , 9 , 3 ] ] NEW_LINE func ( a ) NEW_LINE DEDENT
g = [ dict ( ) for i in range ( 200005 ) ] NEW_LINE s = set ( ) NEW_LINE ns = set ( ) NEW_LINE
def dfs ( x ) : NEW_LINE INDENT global s , g , ns NEW_LINE v = [ ] NEW_LINE v . clear ( ) ; NEW_LINE ns . clear ( ) ; NEW_LINE DEDENT
for it in s : NEW_LINE
if ( x in g and not g [ x ] [ it ] ) : NEW_LINE INDENT v . append ( it ) ; NEW_LINE DEDENT else : NEW_LINE INDENT ns . add ( it ) ; NEW_LINE DEDENT s = ns ; NEW_LINE for i in v : NEW_LINE dfs ( i ) ; NEW_LINE
def weightOfMST ( N ) : NEW_LINE
cnt = 0 ; NEW_LINE
for i in range ( 1 , N + 1 ) : NEW_LINE INDENT s . add ( i ) ; NEW_LINE DEDENT
while ( len ( s ) != 0 ) : NEW_LINE
cnt += 1 NEW_LINE t = list ( s ) [ 0 ] NEW_LINE s . discard ( t ) ; NEW_LINE
dfs ( t ) ; NEW_LINE print ( cnt ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 6 NEW_LINE M = 11 ; NEW_LINE edges = [ [ 1 , 3 ] , [ 1 , 4 ] , [ 1 , 5 ] , [ 1 , 6 ] , [ 2 , 3 ] , [ 2 , 4 ] , [ 2 , 5 ] , [ 2 , 6 ] , [ 3 , 4 ] , [ 3 , 5 ] , [ 3 , 6 ] ] ; NEW_LINE DEDENT
for i in range ( M ) : NEW_LINE INDENT u = edges [ i ] [ 0 ] ; NEW_LINE v = edges [ i ] [ 1 ] ; NEW_LINE g [ u ] [ v ] = 1 ; NEW_LINE g [ v ] [ u ] = 1 ; NEW_LINE DEDENT
weightOfMST ( N ) ; NEW_LINE
def countPairs ( A , B ) : NEW_LINE INDENT n = len ( A ) NEW_LINE A . sort ( ) NEW_LINE B . sort ( ) NEW_LINE ans = 0 NEW_LINE for i in range ( n ) : NEW_LINE INDENT if ( A [ i ] > B [ ans ] ) : NEW_LINE INDENT ans += 1 NEW_LINE DEDENT DEDENT return ans NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT A = [ 30 , 28 , 45 , 22 ] NEW_LINE B = [ 35 , 25 , 22 , 48 ] NEW_LINE print ( countPairs ( A , B ) ) NEW_LINE DEDENT
def maxMod ( arr , n ) : NEW_LINE INDENT maxVal = max ( arr ) NEW_LINE secondMax = 0 NEW_LINE DEDENT
for i in range ( 0 , n ) : NEW_LINE INDENT if ( arr [ i ] < maxVal and arr [ i ] > secondMax ) : NEW_LINE INDENT secondMax = arr [ i ] NEW_LINE DEDENT DEDENT return secondMax NEW_LINE
arr = [ 2 , 4 , 1 , 5 , 3 , 6 ] NEW_LINE n = len ( arr ) NEW_LINE print ( maxMod ( arr , n ) ) NEW_LINE
def isPossible ( A , B , n , m , x , y ) : NEW_LINE
if ( x > n or y > m ) : NEW_LINE INDENT return False NEW_LINE DEDENT
A . sort ( ) NEW_LINE B . sort ( ) NEW_LINE
if ( A [ x - 1 ] < B [ m - y ] ) : NEW_LINE INDENT return True NEW_LINE DEDENT else : NEW_LINE INDENT return False NEW_LINE DEDENT
A = [ 1 , 1 , 1 , 1 , 1 ] NEW_LINE B = [ 2 , 2 ] NEW_LINE n = len ( A ) NEW_LINE m = len ( B ) NEW_LINE x = 3 NEW_LINE y = 1 NEW_LINE if ( isPossible ( A , B , n , m , x , y ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
MAX = 100005 NEW_LINE
def Min_Replace ( arr , n , k ) : NEW_LINE INDENT arr . sort ( reverse = False ) NEW_LINE DEDENT
freq = [ 0 for i in range ( MAX ) ] NEW_LINE p = 0 NEW_LINE freq [ p ] = 1 NEW_LINE
for i in range ( 1 , n , 1 ) : NEW_LINE INDENT if ( arr [ i ] == arr [ i - 1 ] ) : NEW_LINE INDENT freq [ p ] += 1 NEW_LINE DEDENT else : NEW_LINE INDENT p += 1 NEW_LINE freq [ p ] += 1 NEW_LINE DEDENT DEDENT
freq . sort ( reverse = True ) NEW_LINE
ans = 0 NEW_LINE for i in range ( k , p + 1 , 1 ) : NEW_LINE INDENT ans += freq [ i ] NEW_LINE DEDENT
return ans NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 1 , 2 , 7 , 8 , 2 , 3 , 2 , 3 ] NEW_LINE n = len ( arr ) NEW_LINE k = 2 NEW_LINE print ( Min_Replace ( arr , n , k ) ) NEW_LINE DEDENT
def Segment ( x , l , n ) : NEW_LINE
if ( n == 1 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT
ans = 2 NEW_LINE for i in range ( 1 , n - 1 ) : NEW_LINE
if ( x [ i ] - l [ i ] > x [ i - 1 ] ) : NEW_LINE INDENT ans += 1 NEW_LINE DEDENT
elif ( x [ i ] + l [ i ] < x [ i + 1 ] ) : NEW_LINE
x [ i ] = x [ i ] + l [ i ] NEW_LINE ans += 1 NEW_LINE
return ans NEW_LINE
x = [ 1 , 3 , 4 , 5 , 8 ] NEW_LINE l = [ 10 , 1 , 2 , 2 , 5 ] NEW_LINE n = len ( x ) NEW_LINE
print ( Segment ( x , l , n ) ) NEW_LINE
def MinimizeleftOverSum ( a , n ) : NEW_LINE INDENT v1 , v2 = [ ] , [ ] ; NEW_LINE for i in range ( n ) : NEW_LINE INDENT if ( a [ i ] % 2 ) : NEW_LINE INDENT v1 . append ( a [ i ] ) ; NEW_LINE DEDENT else : NEW_LINE INDENT v2 . append ( a [ i ] ) ; NEW_LINE DEDENT DEDENT DEDENT
if ( len ( v1 ) > len ( v2 ) ) : NEW_LINE
v1 . sort ( ) ; NEW_LINE v2 . sort ( ) ; NEW_LINE
x = len ( v1 ) - len ( v2 ) - 1 ; NEW_LINE sum = 0 ; NEW_LINE i = 0 ; NEW_LINE
while ( i < x ) : NEW_LINE INDENT sum += v1 [ i ] ; NEW_LINE i += 1 NEW_LINE DEDENT
return sum ; NEW_LINE
elif ( len ( v2 ) > len ( v1 ) ) : NEW_LINE
v1 . sort ( ) ; NEW_LINE v2 . sort ( ) ; NEW_LINE
x = len ( v2 ) - len ( v1 ) - 1 ; NEW_LINE sum = 0 ; NEW_LINE i = 0 ; NEW_LINE
while ( i < x ) : NEW_LINE INDENT sum += v2 [ i ] ; NEW_LINE i += 1 NEW_LINE DEDENT
return sum ; NEW_LINE
else : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = [ 2 , 2 , 2 , 2 ] ; NEW_LINE n = len ( a ) ; NEW_LINE print ( MinimizeleftOverSum ( a , n ) ) ; NEW_LINE DEDENT
def minOperation ( S , N , K ) : NEW_LINE
if N % K : NEW_LINE INDENT print ( " Not ▁ Possible " ) NEW_LINE return NEW_LINE DEDENT
count = [ 0 ] * 26 NEW_LINE for i in range ( 0 , N ) : NEW_LINE INDENT count [ ord ( S [ i ] ) - 97 ] += 1 NEW_LINE DEDENT E = N // K NEW_LINE greaterE = [ ] NEW_LINE lessE = [ ] NEW_LINE for i in range ( 0 , 26 ) : NEW_LINE
if count [ i ] < E : NEW_LINE INDENT lessE . append ( E - count [ i ] ) NEW_LINE DEDENT else : NEW_LINE INDENT greaterE . append ( count [ i ] - E ) NEW_LINE DEDENT greaterE . sort ( ) NEW_LINE lessE . sort ( ) NEW_LINE mi = float ( ' inf ' ) NEW_LINE for i in range ( 0 , K + 1 ) : NEW_LINE
set1 , set2 = i , K - i NEW_LINE if ( len ( greaterE ) >= set1 and len ( lessE ) >= set2 ) : NEW_LINE INDENT step1 , step2 = 0 , 0 NEW_LINE for j in range ( 0 , set1 ) : NEW_LINE INDENT step1 += greaterE [ j ] NEW_LINE DEDENT for j in range ( 0 , set2 ) : NEW_LINE INDENT step2 += lessE [ j ] NEW_LINE DEDENT mi = min ( mi , max ( step1 , step2 ) ) NEW_LINE DEDENT print ( mi ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT S = " accb " NEW_LINE N = len ( S ) NEW_LINE K = 2 NEW_LINE minOperation ( S , N , K ) NEW_LINE DEDENT
def minMovesToSort ( arr , n ) : NEW_LINE INDENT moves = 0 NEW_LINE mn = arr [ n - 1 ] NEW_LINE for i in range ( n - 1 , - 1 , - 1 ) : NEW_LINE DEDENT
if ( arr [ i ] > mn ) : NEW_LINE INDENT moves += arr [ i ] - mn NEW_LINE DEDENT
return moves NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 3 , 5 , 2 , 8 , 4 ] NEW_LINE n = len ( arr ) NEW_LINE print ( minMovesToSort ( arr , n ) ) NEW_LINE DEDENT
def SieveOfEratosthenes ( n ) : NEW_LINE
prime [ 1 ] = False NEW_LINE p = 2 NEW_LINE while p * p <= n : NEW_LINE
if prime [ p ] : NEW_LINE
for i in range ( p * 2 , n + 1 , p ) : NEW_LINE INDENT prime [ i ] = False NEW_LINE DEDENT p += 1 NEW_LINE
def sortPrimes ( arr , n ) : NEW_LINE INDENT SieveOfEratosthenes ( 100005 ) NEW_LINE DEDENT
v = [ ] NEW_LINE for i in range ( 0 , n ) : NEW_LINE
if prime [ arr [ i ] ] : NEW_LINE INDENT v . append ( arr [ i ] ) NEW_LINE DEDENT v . sort ( reverse = True ) NEW_LINE j = 0 NEW_LINE
for i in range ( 0 , n ) : NEW_LINE INDENT if prime [ arr [ i ] ] : NEW_LINE INDENT arr [ i ] = v [ j ] NEW_LINE j += 1 NEW_LINE DEDENT DEDENT return arr NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 4 , 3 , 2 , 6 , 100 , 17 ] NEW_LINE n = len ( arr ) NEW_LINE prime = [ True ] * 100006 NEW_LINE arr = sortPrimes ( arr , n ) NEW_LINE DEDENT
for i in range ( 0 , n ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT
def findOptimalPairs ( arr , N ) : NEW_LINE INDENT arr . sort ( reverse = False ) NEW_LINE DEDENT
i = 0 NEW_LINE j = N - 1 NEW_LINE while ( i <= j ) : NEW_LINE INDENT print ( " ( " , arr [ i ] , " , " , arr [ j ] , " ) " , end = " ▁ " ) NEW_LINE i += 1 NEW_LINE j -= 1 NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 9 , 6 , 5 , 1 ] NEW_LINE N = len ( arr ) NEW_LINE findOptimalPairs ( arr , N ) NEW_LINE DEDENT
def countBits ( a ) : NEW_LINE INDENT count = 0 NEW_LINE while ( a ) : NEW_LINE INDENT if ( a & 1 ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT a = a >> 1 NEW_LINE DEDENT return count NEW_LINE DEDENT
def insertionSort ( arr , aux , n ) : NEW_LINE INDENT for i in range ( 1 , n , 1 ) : NEW_LINE DEDENT
key1 = aux [ i ] NEW_LINE key2 = arr [ i ] NEW_LINE j = i - 1 NEW_LINE
while ( j >= 0 and aux [ j ] < key1 ) : NEW_LINE INDENT aux [ j + 1 ] = aux [ j ] NEW_LINE arr [ j + 1 ] = arr [ j ] NEW_LINE j = j - 1 NEW_LINE DEDENT aux [ j + 1 ] = key1 NEW_LINE arr [ j + 1 ] = key2 NEW_LINE
def sortBySetBitCount ( arr , n ) : NEW_LINE
aux = [ 0 for i in range ( n ) ] NEW_LINE for i in range ( 0 , n , 1 ) : NEW_LINE INDENT aux [ i ] = countBits ( arr [ i ] ) NEW_LINE DEDENT
insertionSort ( arr , aux , n ) NEW_LINE
def printArr ( arr , n ) : NEW_LINE INDENT for i in range ( 0 , n , 1 ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 1 , 2 , 3 , 4 , 5 , 6 ] NEW_LINE n = len ( arr ) NEW_LINE sortBySetBitCount ( arr , n ) NEW_LINE printArr ( arr , n ) NEW_LINE DEDENT
def countBits ( a ) : NEW_LINE INDENT count = 0 NEW_LINE while ( a ) : NEW_LINE INDENT if ( a & 1 ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT a = a >> 1 NEW_LINE DEDENT return count NEW_LINE DEDENT
def sortBySetBitCount ( arr , n ) : NEW_LINE INDENT count = [ [ ] for i in range ( 32 ) ] NEW_LINE setbitcount = 0 NEW_LINE for i in range ( n ) : NEW_LINE INDENT setbitcount = countBits ( arr [ i ] ) NEW_LINE count [ setbitcount ] . append ( arr [ i ] ) NEW_LINE DEDENT DEDENT
for i in range ( 31 , - 1 , - 1 ) : NEW_LINE INDENT v1 = count [ i ] NEW_LINE for i in range ( len ( v1 ) ) : NEW_LINE INDENT arr [ j ] = v1 [ i ] NEW_LINE j += 1 NEW_LINE DEDENT DEDENT
def printArr ( arr , n ) : NEW_LINE INDENT print ( * arr ) NEW_LINE DEDENT
arr = [ 1 , 2 , 3 , 4 , 5 , 6 ] NEW_LINE n = len ( arr ) NEW_LINE sortBySetBitCount ( arr , n ) NEW_LINE printArr ( arr , n ) NEW_LINE
def generateString ( k1 , k2 , s ) : NEW_LINE
s = list ( s ) NEW_LINE C1s = 0 NEW_LINE C0s = 0 NEW_LINE flag = 0 NEW_LINE pos = [ ] NEW_LINE
for i in range ( len ( s ) ) : NEW_LINE INDENT if ( s [ i ] == '0' ) : NEW_LINE INDENT C0s += 1 NEW_LINE DEDENT DEDENT
if ( ( i + 1 ) % k1 != 0 and ( i + 1 ) % k2 != 0 ) : NEW_LINE INDENT pos . append ( i ) NEW_LINE DEDENT else : NEW_LINE C1s += 1 NEW_LINE if ( C0s >= C1s ) : NEW_LINE
if ( len ( pos ) == 0 ) : NEW_LINE INDENT print ( - 1 ) NEW_LINE flag = 1 NEW_LINE break NEW_LINE DEDENT
else : NEW_LINE INDENT k = pos [ len ( pos ) - 1 ] NEW_LINE s [ k ] = '1' NEW_LINE C0s -= 1 NEW_LINE C1s += 1 NEW_LINE pos = pos [ : - 1 ] NEW_LINE DEDENT
s = ' ' . join ( s ) NEW_LINE if ( flag == 0 ) : NEW_LINE INDENT print ( s ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT K1 = 2 NEW_LINE K2 = 4 NEW_LINE S = "11000100" NEW_LINE generateString ( K1 , K2 , S ) NEW_LINE DEDENT
import math NEW_LINE
def maximizeProduct ( N ) : NEW_LINE
MSB = ( int ) ( math . log2 ( N ) ) NEW_LINE
X = 1 << MSB NEW_LINE
Y = N - ( 1 << MSB ) NEW_LINE
for i in range ( MSB ) : NEW_LINE
if ( not ( N & ( 1 << i ) ) ) : NEW_LINE
X += 1 << i NEW_LINE
Y += 1 << i NEW_LINE
print ( X , Y ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT N = 45 NEW_LINE maximizeProduct ( N ) NEW_LINE DEDENT
from math import log10 NEW_LINE
def check ( num ) : NEW_LINE
sm = 0 NEW_LINE
num2 = num * num NEW_LINE while ( num ) : NEW_LINE INDENT sm += num % 10 NEW_LINE num //= 10 NEW_LINE DEDENT
sm2 = 0 NEW_LINE while ( num2 ) : NEW_LINE INDENT sm2 += num2 % 10 NEW_LINE num2 //= 10 NEW_LINE DEDENT return ( ( sm * sm ) == sm2 ) NEW_LINE
def convert ( s ) : NEW_LINE INDENT val = 0 NEW_LINE s = s [ : : - 1 ] NEW_LINE cur = 1 NEW_LINE for i in range ( len ( s ) ) : NEW_LINE INDENT val += ( ord ( s [ i ] ) - ord ( '0' ) ) * cur NEW_LINE cur *= 10 NEW_LINE DEDENT return val NEW_LINE DEDENT
def generate ( s , len1 , uniq ) : NEW_LINE
if ( len ( s ) == len1 ) : NEW_LINE
if ( check ( convert ( s ) ) ) : NEW_LINE INDENT uniq . add ( convert ( s ) ) NEW_LINE DEDENT return NEW_LINE
for i in range ( 4 ) : NEW_LINE INDENT generate ( s + chr ( i + ord ( '0' ) ) , len1 , uniq ) NEW_LINE DEDENT
def totalNumbers ( L , R ) : NEW_LINE
ans = 0 NEW_LINE
max_len = int ( log10 ( R ) ) + 1 NEW_LINE
uniq = set ( ) NEW_LINE for i in range ( 1 , max_len + 1 , 1 ) : NEW_LINE
generate ( " " , i , uniq ) NEW_LINE
for x in uniq : NEW_LINE INDENT if ( x >= L and x <= R ) : NEW_LINE INDENT ans += 1 NEW_LINE DEDENT DEDENT return ans NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT L = 22 NEW_LINE R = 22 NEW_LINE print ( totalNumbers ( L , R ) ) NEW_LINE DEDENT
def convertXintoY ( X , Y ) : NEW_LINE
while ( Y > X ) : NEW_LINE
if ( Y % 2 == 0 ) : NEW_LINE INDENT Y //= 2 NEW_LINE DEDENT
elif ( Y % 10 == 1 ) : NEW_LINE INDENT Y //= 10 NEW_LINE DEDENT
else : NEW_LINE INDENT break NEW_LINE DEDENT
if ( X == Y ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT X , Y = 100 , 40021 NEW_LINE convertXintoY ( X , Y ) NEW_LINE DEDENT
def generateString ( K ) : NEW_LINE
s = " " NEW_LINE
for i in range ( 97 , 97 + K , 1 ) : NEW_LINE INDENT s = s + chr ( i ) ; NEW_LINE DEDENT
for j in range ( i + 1 , 97 + K , 1 ) : NEW_LINE INDENT s += chr ( i ) NEW_LINE s += chr ( j ) NEW_LINE DEDENT
s += chr ( 97 ) NEW_LINE
print ( s ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT K = 4 NEW_LINE generateString ( K ) NEW_LINE DEDENT
def findEquation ( S , M ) : NEW_LINE
print ( "1 ▁ " , ( ( - 1 ) * S ) , " ▁ " , M ) NEW_LINE
S = 5 NEW_LINE M = 6 NEW_LINE findEquation ( S , M ) NEW_LINE
def minSteps ( a , n ) : NEW_LINE
prefix_sum = a [ : ] NEW_LINE
for i in range ( 1 , n ) : NEW_LINE INDENT prefix_sum [ i ] += prefix_sum [ i - 1 ] NEW_LINE DEDENT
mx = - 1 NEW_LINE
for subgroupsum in prefix_sum : NEW_LINE INDENT sum = 0 NEW_LINE i = 0 NEW_LINE grp_count = 0 NEW_LINE DEDENT
while i < n : NEW_LINE INDENT sum += a [ i ] NEW_LINE DEDENT
if sum == subgroupsum : NEW_LINE
grp_count += 1 NEW_LINE sum = 0 NEW_LINE
elif sum > subgroupsum : NEW_LINE INDENT grp_count = - 1 NEW_LINE break NEW_LINE DEDENT i += 1 NEW_LINE
if grp_count > mx : NEW_LINE INDENT mx = grp_count NEW_LINE DEDENT
return n - mx NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT A = [ 1 , 2 , 3 , 2 , 1 , 3 ] NEW_LINE N = len ( A ) NEW_LINE DEDENT
print ( minSteps ( A , N ) ) NEW_LINE
def maxOccuringCharacter ( s ) : NEW_LINE
INDENT count0 = 0 NEW_LINE count1 = 0 NEW_LINE DEDENT
INDENT for i in range ( len ( s ) ) : NEW_LINE DEDENT
if ( s [ i ] == '1' ) : NEW_LINE count1 += 1 NEW_LINE
elif ( s [ i ] == '0' ) : NEW_LINE count0 += 1 NEW_LINE
INDENT prev = - 1 NEW_LINE for i in range ( len ( s ) ) : NEW_LINE INDENT if ( s [ i ] == '1' ) : NEW_LINE prev = i NEW_LINE break NEW_LINE DEDENT DEDENT
INDENT for i in range ( prev + 1 , len ( s ) ) : NEW_LINE DEDENT
if ( s [ i ] != ' X ' ) : NEW_LINE
if ( s [ i ] == '1' ) : NEW_LINE INDENT count1 += i - prev - 1 NEW_LINE prev = i NEW_LINE DEDENT
else : NEW_LINE
flag = True NEW_LINE for j in range ( i + 1 , len ( s ) ) : NEW_LINE if ( s [ j ] == '1' ) : NEW_LINE INDENT flag = False NEW_LINE prev = j NEW_LINE break NEW_LINE DEDENT
if ( flag == False ) : NEW_LINE i = prev NEW_LINE
else : NEW_LINE i = len ( s ) NEW_LINE
INDENT prev = - 1 NEW_LINE for i in range ( 0 , len ( s ) ) : NEW_LINE INDENT if ( s [ i ] == '0' ) : NEW_LINE prev = i NEW_LINE break NEW_LINE DEDENT DEDENT
INDENT for i in range ( prev + 1 , len ( s ) ) : NEW_LINE DEDENT
if ( s [ i ] != ' X ' ) : NEW_LINE
if ( s [ i ] == '0' ) : NEW_LINE
count0 += i - prev - 1 NEW_LINE
prev = i NEW_LINE
else : NEW_LINE
flag = True NEW_LINE for j in range ( i + 1 , len ( s ) ) : NEW_LINE if ( s [ j ] == '0' ) : NEW_LINE INDENT prev = j NEW_LINE flag = False NEW_LINE break NEW_LINE DEDENT
if ( flag == False ) : NEW_LINE i = prev NEW_LINE
else : NEW_LINE i = len ( s ) NEW_LINE
INDENT if ( s [ 0 ] == ' X ' ) : NEW_LINE DEDENT
count = 0 NEW_LINE i = 0 NEW_LINE while ( s [ i ] == ' X ' ) : NEW_LINE count += 1 NEW_LINE i += 1 NEW_LINE
if ( s [ i ] == '1' ) : NEW_LINE count1 += count NEW_LINE
INDENT if ( s [ ( len ( s ) - 1 ) ] == ' X ' ) : NEW_LINE DEDENT
count = 0 NEW_LINE i = len ( s ) - 1 NEW_LINE while ( s [ i ] == ' X ' ) : NEW_LINE count += 1 NEW_LINE i -= 1 NEW_LINE
if ( s [ i ] == '0' ) : NEW_LINE count0 += count NEW_LINE
INDENT if ( count0 == count1 ) : NEW_LINE INDENT print ( " X " ) NEW_LINE DEDENT DEDENT
INDENT elif ( count0 > count1 ) : NEW_LINE INDENT print ( 0 ) NEW_LINE DEDENT DEDENT
INDENT else : NEW_LINE INDENT print ( 1 ) NEW_LINE DEDENT DEDENT
S = " XX10XX10XXX1XX " NEW_LINE maxOccuringCharacter ( S ) NEW_LINE
def maxSheets ( A , B ) : NEW_LINE INDENT area = A * B NEW_LINE DEDENT
count = 1 NEW_LINE
while ( area % 2 == 0 ) : NEW_LINE
area //= 2 NEW_LINE
count *= 2 NEW_LINE return count NEW_LINE
A = 5 NEW_LINE B = 10 NEW_LINE print ( maxSheets ( A , B ) ) NEW_LINE
def findMinMoves ( a , b ) : NEW_LINE
ans = 0 NEW_LINE
if ( a == b or abs ( a - b ) == 1 ) : NEW_LINE INDENT ans = a + b NEW_LINE DEDENT else : NEW_LINE
k = min ( a , b ) NEW_LINE
j = max ( a , b ) NEW_LINE ans = 2 * k + 2 * ( j - k ) - 1 NEW_LINE
print ( ans ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
a , b = 3 , 5 NEW_LINE
findMinMoves ( a , b ) NEW_LINE
def cntEvenSumPairs ( X , Y ) : NEW_LINE
cntXEvenNums = X / 2 NEW_LINE
cntXOddNums = ( X + 1 ) / 2 NEW_LINE
cntYEvenNums = Y / 2 NEW_LINE
cntYOddNums = ( Y + 1 ) / 2 NEW_LINE
cntPairs = ( ( cntXEvenNums * cntYEvenNums ) + ( cntXOddNums * cntYOddNums ) ) NEW_LINE
return cntPairs NEW_LINE
X = 2 NEW_LINE Y = 3 NEW_LINE print ( cntEvenSumPairs ( X , Y ) ) NEW_LINE
import sys NEW_LINE
def minMoves ( arr ) : NEW_LINE INDENT N = len ( arr ) NEW_LINE DEDENT
if ( N <= 2 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
ans = sys . maxsize NEW_LINE
for i in range ( - 1 , 2 ) : NEW_LINE INDENT for j in range ( - 1 , 2 ) : NEW_LINE DEDENT
num1 = arr [ 0 ] + i NEW_LINE
num2 = arr [ 1 ] + j NEW_LINE flag = 1 NEW_LINE moves = abs ( i ) + abs ( j ) NEW_LINE
for idx in range ( 2 , N ) : NEW_LINE
num = num1 + num2 NEW_LINE
if ( abs ( arr [ idx ] - num ) > 1 ) : NEW_LINE INDENT flag = 0 NEW_LINE DEDENT
else : NEW_LINE INDENT moves += abs ( arr [ idx ] - num ) NEW_LINE DEDENT num1 = num2 NEW_LINE num2 = num NEW_LINE
if ( flag ) : NEW_LINE INDENT ans = min ( ans , moves ) NEW_LINE DEDENT
if ( ans == sys . maxsize ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT return ans NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 4 , 8 , 9 , 17 , 27 ] NEW_LINE print ( minMoves ( arr ) ) NEW_LINE DEDENT
def querySum ( arr , N , Q , M ) : NEW_LINE
for i in range ( M ) : NEW_LINE INDENT x = Q [ i ] [ 0 ] NEW_LINE y = Q [ i ] [ 1 ] NEW_LINE DEDENT
sum = 0 NEW_LINE
while ( x < N ) : NEW_LINE
sum += arr [ x ] NEW_LINE
x += y NEW_LINE print ( sum , end = " ▁ " ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 1 , 2 , 7 , 5 , 4 ] ; NEW_LINE Q = [ [ 2 , 1 ] , [ 3 , 2 ] ] NEW_LINE N = len ( arr ) NEW_LINE M = len ( Q ) NEW_LINE querySum ( arr , N , Q , M ) NEW_LINE DEDENT
def findBitwiseORGivenXORAND ( X , Y ) : NEW_LINE INDENT return X + Y NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT X = 5 NEW_LINE Y = 2 NEW_LINE print ( findBitwiseORGivenXORAND ( X , Y ) ) NEW_LINE DEDENT
def GCD ( a , b ) : NEW_LINE
if ( b == 0 ) : NEW_LINE INDENT return a NEW_LINE DEDENT
return GCD ( b , a % b ) NEW_LINE
def canReach ( N , A , B , K ) : NEW_LINE
gcd = GCD ( N , K ) NEW_LINE
if ( abs ( A - B ) % gcd == 0 ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT
else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 5 NEW_LINE A = 2 NEW_LINE B = 1 NEW_LINE K = 2 NEW_LINE DEDENT
canReach ( N , A , B , K ) NEW_LINE
from collections import defaultdict NEW_LINE
def countOfSubarray ( arr , N ) : NEW_LINE
mp = defaultdict ( lambda : 0 ) NEW_LINE
answer = 0 NEW_LINE
sum = 0 NEW_LINE
mp [ 1 ] += 1 NEW_LINE
for i in range ( N ) : NEW_LINE
sum += arr [ i ] NEW_LINE answer += mp [ sum - i ] NEW_LINE
mp [ sum - i ] += 1 NEW_LINE
print ( answer ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
arr = [ 1 , 0 , 2 , 1 , 2 , - 2 , 2 , 4 ] NEW_LINE
N = len ( arr ) NEW_LINE
countOfSubarray ( arr , N ) NEW_LINE
def minAbsDiff ( N ) : NEW_LINE
sumSet1 = 0 NEW_LINE
sumSet2 = 0 NEW_LINE
for i in reversed ( range ( N + 1 ) ) : NEW_LINE
if sumSet1 <= sumSet2 : NEW_LINE sumSet1 = sumSet1 + i NEW_LINE else : NEW_LINE sumSet2 = sumSet2 + i NEW_LINE return abs ( sumSet1 - sumSet2 ) NEW_LINE
N = 6 NEW_LINE print ( minAbsDiff ( N ) ) NEW_LINE
def checkDigits ( n ) : NEW_LINE
while True : NEW_LINE INDENT r = n % 10 NEW_LINE DEDENT
if ( r == 3 or r == 4 or r == 6 or r == 7 or r == 9 ) : NEW_LINE INDENT return False NEW_LINE DEDENT n //= 10 NEW_LINE if n == 0 : NEW_LINE INDENT break NEW_LINE DEDENT return True NEW_LINE
def isPrime ( n ) : NEW_LINE INDENT if ( n <= 1 ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT
for i in range ( 2 , n + 1 ) : NEW_LINE INDENT if i * i > n : NEW_LINE INDENT break NEW_LINE DEDENT if ( n % i == 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT return True NEW_LINE
def isAllPrime ( n ) : NEW_LINE INDENT return isPrime ( n ) and checkDigits ( n ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 101 NEW_LINE if ( isAllPrime ( N ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT DEDENT
def minCost ( str , a , b ) : NEW_LINE
openUnbalanced = 0 ; NEW_LINE
closedUnbalanced = 0 ; NEW_LINE
openCount = 0 ; NEW_LINE
closedCount = 0 ; NEW_LINE for i in range ( len ( str ) ) : NEW_LINE
if ( str [ i ] == ' ( ' ) : NEW_LINE INDENT openUnbalanced += 1 ; NEW_LINE openCount += 1 ; NEW_LINE DEDENT
else : NEW_LINE
if ( openUnbalanced == 0 ) : NEW_LINE
closedUnbalanced += 1 ; NEW_LINE
else : NEW_LINE
openUnbalanced -= 1 ; NEW_LINE
closedCount += 1 ; NEW_LINE
result = a * ( abs ( openCount - closedCount ) ) ; NEW_LINE
if ( closedCount > openCount ) : NEW_LINE INDENT closedUnbalanced -= ( closedCount - openCount ) ; NEW_LINE DEDENT if ( openCount > closedCount ) : NEW_LINE INDENT openUnbalanced -= ( openCount - closedCount ) ; NEW_LINE DEDENT
result += min ( a * ( openUnbalanced + closedUnbalanced ) , b * closedUnbalanced ) ; NEW_LINE
print ( result ) ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : str = " ) ) ( ) ( ( ) ( ) ( " ; NEW_LINE INDENT A = 1 ; B = 3 ; NEW_LINE minCost ( str , A , B ) ; NEW_LINE DEDENT
def countEvenSum ( low , high , k ) : NEW_LINE
even_count = high / 2 - ( low - 1 ) / 2 NEW_LINE odd_count = ( high + 1 ) / 2 - low / 2 NEW_LINE even_sum = 1 NEW_LINE odd_sum = 0 NEW_LINE
for i in range ( 0 , k ) : NEW_LINE
prev_even = even_sum NEW_LINE prev_odd = odd_sum NEW_LINE
even_sum = ( ( prev_even * even_count ) + ( prev_odd * odd_count ) ) NEW_LINE
odd_sum = ( ( prev_even * odd_count ) + ( prev_odd * even_count ) ) NEW_LINE
print ( int ( even_sum ) ) NEW_LINE
low = 4 ; NEW_LINE high = 5 ; NEW_LINE
K = 3 ; NEW_LINE
countEvenSum ( low , high , K ) ; NEW_LINE
def count ( n , k ) : NEW_LINE INDENT count = ( pow ( 10 , k ) - pow ( 10 , k - 1 ) ) ; NEW_LINE DEDENT
print ( count ) ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 2 ; NEW_LINE k = 1 ; NEW_LINE count ( n , k ) ; NEW_LINE DEDENT
def func ( N , P ) : NEW_LINE
sumUptoN = ( N * ( N + 1 ) / 2 ) ; NEW_LINE sumOfMultiplesOfP = 0 ; NEW_LINE
if ( N < P ) : NEW_LINE INDENT return sumUptoN ; NEW_LINE DEDENT
elif ( ( N / P ) == 1 ) : NEW_LINE INDENT return sumUptoN - P + 1 ; NEW_LINE DEDENT
sumOfMultiplesOfP = ( ( ( N / P ) * ( 2 * P + ( N / P - 1 ) * P ) ) / 2 ) ; NEW_LINE
return ( sumUptoN + func ( N / P , P ) - sumOfMultiplesOfP ) ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
N = 10 ; NEW_LINE P = 5 ; NEW_LINE
print ( func ( N , P ) ) ; NEW_LINE
def findShifts ( A , N ) : NEW_LINE
shift = [ 0 for i in range ( N ) ] NEW_LINE for i in range ( N ) : NEW_LINE
if ( i == A [ i ] - 1 ) : NEW_LINE INDENT shift [ i ] = 0 NEW_LINE DEDENT
else : NEW_LINE
shift [ i ] = ( A [ i ] - 1 - i + N ) % N NEW_LINE
for i in range ( N ) : NEW_LINE INDENT print ( shift [ i ] , end = " ▁ " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 1 , 4 , 3 , 2 , 5 ] NEW_LINE N = len ( arr ) NEW_LINE findShifts ( arr , N ) NEW_LINE DEDENT
def constructmatrix ( N ) : NEW_LINE INDENT check = bool ( True ) NEW_LINE for i in range ( N ) : NEW_LINE INDENT for j in range ( N ) : NEW_LINE DEDENT DEDENT
if ( i == j ) : NEW_LINE INDENT print ( 1 , end = " ▁ " ) NEW_LINE DEDENT elif ( check ) : NEW_LINE
print ( 2 , end = " ▁ " ) NEW_LINE check = bool ( False ) NEW_LINE else : NEW_LINE
print ( - 2 , end = " ▁ " ) NEW_LINE check = bool ( True ) NEW_LINE print ( ) NEW_LINE
N = 5 NEW_LINE constructmatrix ( 5 ) NEW_LINE
def check ( unit_digit , X ) : NEW_LINE
for times in range ( 1 , 11 ) : NEW_LINE INDENT digit = ( X * times ) % 10 NEW_LINE if ( digit == unit_digit ) : NEW_LINE INDENT return times NEW_LINE DEDENT DEDENT
return - 1 NEW_LINE
def getNum ( N , X ) : NEW_LINE
unit_digit = N % 10 NEW_LINE
times = check ( unit_digit , X ) NEW_LINE
if ( times == - 1 ) : NEW_LINE INDENT return times NEW_LINE DEDENT
else : NEW_LINE
if ( N >= ( times * X ) ) : NEW_LINE
return times NEW_LINE
else : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
N = 58 NEW_LINE X = 7 NEW_LINE print ( getNum ( N , X ) ) NEW_LINE
def minPoints ( n , m ) : NEW_LINE INDENT ans = 0 NEW_LINE DEDENT
if ( ( n % 2 != 0 ) and ( m % 2 != 0 ) ) : NEW_LINE INDENT ans = ( ( n * m ) // 2 ) + 1 NEW_LINE DEDENT else : NEW_LINE INDENT ans = ( n * m ) // 2 NEW_LINE DEDENT
return ans NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
N = 5 NEW_LINE M = 7 NEW_LINE
print ( minPoints ( N , M ) ) NEW_LINE
def getLargestString ( s , k ) : NEW_LINE
frequency_array = [ 0 ] * 26 NEW_LINE
for i in range ( len ( s ) ) : NEW_LINE INDENT frequency_array [ ord ( s [ i ] ) - ord ( ' a ' ) ] += 1 NEW_LINE DEDENT
ans = " " NEW_LINE
i = 25 NEW_LINE while i >= 0 : NEW_LINE
if ( frequency_array [ i ] > k ) : NEW_LINE
temp = k NEW_LINE st = chr ( i + ord ( ' a ' ) ) NEW_LINE while ( temp > 0 ) : NEW_LINE
ans += st NEW_LINE temp -= 1 NEW_LINE frequency_array [ i ] -= k NEW_LINE
j = i - 1 NEW_LINE while ( frequency_array [ j ] <= 0 and j >= 0 ) : NEW_LINE INDENT j -= 1 NEW_LINE DEDENT
if ( frequency_array [ j ] > 0 and j >= 0 ) : NEW_LINE INDENT str1 = chr ( j + ord ( ' a ' ) ) NEW_LINE ans += str1 NEW_LINE frequency_array [ j ] -= 1 NEW_LINE DEDENT else : NEW_LINE
break NEW_LINE
elif ( frequency_array [ i ] > 0 ) : NEW_LINE
temp = frequency_array [ i ] NEW_LINE frequency_array [ i ] -= temp NEW_LINE st = chr ( i + ord ( ' a ' ) ) NEW_LINE while ( temp > 0 ) : NEW_LINE INDENT ans += st NEW_LINE temp -= 1 NEW_LINE DEDENT
else : NEW_LINE INDENT i -= 1 NEW_LINE DEDENT return ans NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT S = " xxxxzza " NEW_LINE k = 3 NEW_LINE print ( getLargestString ( S , k ) ) NEW_LINE DEDENT
def minOperations ( a , b , n ) : NEW_LINE
minA = min ( a ) ; NEW_LINE
for x in range ( minA , - 1 , - 1 ) : NEW_LINE
check = True ; NEW_LINE
operations = 0 ; NEW_LINE
for i in range ( n ) : NEW_LINE INDENT if ( x % b [ i ] == a [ i ] % b [ i ] ) : NEW_LINE INDENT operations += ( a [ i ] - x ) / b [ i ] ; NEW_LINE DEDENT DEDENT
else : NEW_LINE INDENT check = False ; NEW_LINE break ; NEW_LINE DEDENT if ( check ) : NEW_LINE return operations ; NEW_LINE return - 1 ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 5 ; NEW_LINE A = [ 5 , 7 , 10 , 5 , 15 ] ; NEW_LINE B = [ 2 , 2 , 1 , 3 , 5 ] ; NEW_LINE print ( int ( minOperations ( A , B , N ) ) ) ; NEW_LINE DEDENT
def getLargestSum ( N ) : NEW_LINE
max_sum = 0 NEW_LINE
for i in range ( 1 , N + 1 ) : NEW_LINE INDENT for j in range ( i + 1 , N + 1 , 1 ) : NEW_LINE DEDENT
if ( i * j % ( i + j ) == 0 ) : NEW_LINE
max_sum = max ( max_sum , i + j ) NEW_LINE
return max_sum NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 25 NEW_LINE max_sum = getLargestSum ( N ) NEW_LINE print ( max_sum ) NEW_LINE DEDENT
def maxSubArraySum ( a , size ) : NEW_LINE INDENT max_so_far = - 10 ** 9 NEW_LINE max_ending_here = 0 NEW_LINE DEDENT
for i in range ( size ) : NEW_LINE INDENT max_ending_here = max_ending_here + a [ i ] NEW_LINE if ( max_ending_here < 0 ) : NEW_LINE INDENT max_ending_here = 0 NEW_LINE DEDENT if ( max_so_far < max_ending_here ) : NEW_LINE INDENT max_so_far = max_ending_here NEW_LINE DEDENT DEDENT return max_so_far NEW_LINE
def maxSum ( a , n ) : NEW_LINE
S = 0 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT S += a [ i ] NEW_LINE DEDENT X = maxSubArraySum ( a , n ) NEW_LINE
return 2 * X - S NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT a = [ - 1 , - 2 , - 3 ] NEW_LINE n = len ( a ) NEW_LINE max_sum = maxSum ( a , n ) NEW_LINE print ( max_sum ) NEW_LINE DEDENT
import math NEW_LINE
def isPrime ( n ) : NEW_LINE INDENT flag = 1 NEW_LINE DEDENT
i = 2 NEW_LINE while ( i * i <= n ) : NEW_LINE INDENT if ( n % i == 0 ) : NEW_LINE INDENT flag = 0 NEW_LINE break NEW_LINE DEDENT i += 1 NEW_LINE DEDENT return ( True if flag == 1 else False ) NEW_LINE
def isPerfectSquare ( x ) : NEW_LINE
sr = math . sqrt ( x ) NEW_LINE
return ( ( sr - math . floor ( sr ) ) == 0 ) NEW_LINE
def countInterestingPrimes ( n ) : NEW_LINE INDENT answer = 0 NEW_LINE for i in range ( 2 , n ) : NEW_LINE DEDENT
if ( isPrime ( i ) ) : NEW_LINE
j = 1 NEW_LINE while ( j * j * j * j <= i ) : NEW_LINE
if ( isPerfectSquare ( i - j * j * j * j ) ) : NEW_LINE INDENT answer += 1 NEW_LINE break NEW_LINE DEDENT j += 1 NEW_LINE
return answer NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 10 NEW_LINE print ( countInterestingPrimes ( N ) ) NEW_LINE DEDENT
import math NEW_LINE
def decBinary ( arr , n ) : NEW_LINE INDENT k = int ( math . log2 ( n ) ) NEW_LINE while ( n > 0 ) : NEW_LINE INDENT arr [ k ] = n % 2 NEW_LINE k = k - 1 NEW_LINE n = n // 2 NEW_LINE DEDENT DEDENT
def binaryDec ( arr , n ) : NEW_LINE INDENT ans = 0 NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT ans = ans + ( arr [ i ] << ( n - i - 1 ) ) NEW_LINE DEDENT return ans NEW_LINE DEDENT
def maxNum ( n , k ) : NEW_LINE
l = int ( math . log2 ( n ) ) + 1 NEW_LINE
a = [ 0 for i in range ( 0 , l ) ] NEW_LINE decBinary ( a , n ) NEW_LINE
cn = 0 NEW_LINE for i in range ( 0 , l ) : NEW_LINE INDENT if ( a [ i ] == 0 and cn < k ) : NEW_LINE INDENT a [ i ] = 1 NEW_LINE cn = cn + 1 NEW_LINE DEDENT DEDENT
return binaryDec ( a , l ) NEW_LINE
n = 4 NEW_LINE k = 1 NEW_LINE print ( maxNum ( n , k ) ) NEW_LINE
def findSubSeq ( arr , n , sum ) : NEW_LINE INDENT for i in range ( n - 1 , - 1 , - 1 ) : NEW_LINE DEDENT
if ( sum < arr [ i ] ) : NEW_LINE INDENT arr [ i ] = - 1 ; NEW_LINE DEDENT
else : NEW_LINE INDENT sum -= arr [ i ] ; NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE
if ( arr [ i ] != - 1 ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) ; NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 17 , 25 , 46 , 94 , 201 , 400 ] ; NEW_LINE n = len ( arr ) ; NEW_LINE sum = 272 ; NEW_LINE findSubSeq ( arr , n , sum ) ; NEW_LINE DEDENT
MAX = 26 NEW_LINE
def maxAlpha ( str , len ) : NEW_LINE
first = [ - 1 for x in range ( MAX ) ] NEW_LINE last = [ - 1 for x in range ( MAX ) ] NEW_LINE
for i in range ( 0 , len ) : NEW_LINE INDENT index = ord ( str [ i ] ) - 97 NEW_LINE DEDENT
if ( first [ index ] == - 1 ) : NEW_LINE INDENT first [ index ] = i NEW_LINE DEDENT last [ index ] = i NEW_LINE
ans = - 1 NEW_LINE maxVal = - 1 NEW_LINE
for i in range ( 0 , MAX ) : NEW_LINE
if ( first [ i ] == - 1 ) : NEW_LINE INDENT continue NEW_LINE DEDENT
if ( ( last [ i ] - first [ i ] ) > maxVal ) : NEW_LINE INDENT maxVal = last [ i ] - first [ i ] ; NEW_LINE ans = i NEW_LINE DEDENT return chr ( ans + 97 ) NEW_LINE
str = " abbba " NEW_LINE len = len ( str ) NEW_LINE print ( maxAlpha ( str , len ) ) NEW_LINE
MAX = 100001 ; NEW_LINE
def find_distinct ( a , n , q , queries ) : NEW_LINE INDENT check = [ 0 ] * MAX ; NEW_LINE idx = [ 0 ] * MAX ; NEW_LINE cnt = 1 ; NEW_LINE for i in range ( n - 1 , - 1 , - 1 ) : NEW_LINE DEDENT
if ( check [ a [ i ] ] == 0 ) : NEW_LINE
idx [ i ] = cnt ; NEW_LINE check [ a [ i ] ] = 1 ; NEW_LINE cnt += 1 ; NEW_LINE else : NEW_LINE
idx [ i ] = cnt - 1 ; NEW_LINE
for i in range ( 0 , q ) : NEW_LINE INDENT m = queries [ i ] ; NEW_LINE print ( idx [ m ] , end = " ▁ " ) ; NEW_LINE DEDENT
a = [ 1 , 2 , 3 , 1 , 2 , 3 , 4 , 5 ] ; NEW_LINE n = len ( a ) ; NEW_LINE queries = [ 0 , 3 , 5 , 7 ] ; NEW_LINE q = len ( queries ) ; NEW_LINE find_distinct ( a , n , q , queries ) ; NEW_LINE
MAX = 24 ; NEW_LINE
def countOp ( x ) : NEW_LINE
arr = [ 0 ] * MAX ; NEW_LINE arr [ 0 ] = 1 ; NEW_LINE for i in range ( 1 , MAX ) : NEW_LINE INDENT arr [ i ] = arr [ i - 1 ] * 2 ; NEW_LINE DEDENT
temp = x ; NEW_LINE flag = True ; NEW_LINE
ans = 0 ; NEW_LINE
operations = 0 ; NEW_LINE flag2 = False ; NEW_LINE for i in range ( MAX ) : NEW_LINE INDENT if ( arr [ i ] - 1 == x ) : NEW_LINE INDENT flag2 = True ; NEW_LINE DEDENT DEDENT
if ( arr [ i ] > x ) : NEW_LINE INDENT ans = i ; NEW_LINE break ; NEW_LINE DEDENT
if ( flag2 ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT while ( flag ) : NEW_LINE
if ( arr [ ans ] < x ) : NEW_LINE INDENT ans += 1 ; NEW_LINE DEDENT operations += 1 ; NEW_LINE
for i in range ( MAX ) : NEW_LINE INDENT take = x ^ ( arr [ i ] - 1 ) ; NEW_LINE if ( take <= arr [ ans ] - 1 ) : NEW_LINE DEDENT
if ( take > temp ) : NEW_LINE INDENT temp = take ; NEW_LINE DEDENT
if ( temp == arr [ ans ] - 1 ) : NEW_LINE INDENT flag = False ; NEW_LINE break ; NEW_LINE DEDENT temp += 1 ; NEW_LINE operations += 1 ; NEW_LINE x = temp ; NEW_LINE if ( x == arr [ ans ] - 1 ) : NEW_LINE INDENT flag = False ; NEW_LINE DEDENT
return operations ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT x = 39 ; NEW_LINE print ( countOp ( x ) ) ; NEW_LINE DEDENT
def minOperations ( arr , n ) : NEW_LINE INDENT result = 0 NEW_LINE DEDENT
freq = [ 0 ] * 1000001 NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT freq [ arr [ i ] ] += 1 NEW_LINE DEDENT
maxi = max ( arr ) NEW_LINE for i in range ( 1 , maxi + 1 ) : NEW_LINE INDENT if freq [ i ] != 0 : NEW_LINE DEDENT
for j in range ( i * 2 , maxi + 1 , i ) : NEW_LINE
freq [ j ] = 0 NEW_LINE
result += 1 NEW_LINE return result NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 2 , 4 , 2 , 4 , 4 , 4 ] NEW_LINE n = len ( arr ) NEW_LINE print ( minOperations ( arr , n ) ) NEW_LINE DEDENT
from math import gcd NEW_LINE
def minGCD ( arr , n ) : NEW_LINE INDENT minGCD = 0 ; NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT minGCD = gcd ( minGCD , arr [ i ] ) ; NEW_LINE DEDENT return minGCD ; NEW_LINE
def minLCM ( arr , n ) : NEW_LINE INDENT minLCM = arr [ 0 ] ; NEW_LINE DEDENT
for i in range ( 1 , n ) : NEW_LINE INDENT minLCM = min ( minLCM , arr [ i ] ) ; NEW_LINE DEDENT return minLCM ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 2 , 66 , 14 , 521 ] ; NEW_LINE n = len ( arr ) ; NEW_LINE print ( " LCM ▁ = ▁ " , minLCM ( arr , n ) , " , ▁ GCD ▁ = " , minGCD ( arr , n ) ) ; NEW_LINE DEDENT
import math NEW_LINE
def formStringMinOperations ( ss ) : NEW_LINE
count = [ 0 ] * 3 ; NEW_LINE s = list ( ss ) ; NEW_LINE for i in range ( len ( s ) ) : NEW_LINE INDENT count [ ord ( s [ i ] ) - ord ( '0' ) ] += 1 ; NEW_LINE DEDENT
processed = [ 0 ] * 3 ; NEW_LINE
reqd = math . floor ( len ( s ) / 3 ) ; NEW_LINE for i in range ( len ( s ) ) : NEW_LINE
if ( count [ ord ( s [ i ] ) - ord ( '0' ) ] == reqd ) : NEW_LINE INDENT continue ; NEW_LINE DEDENT
if ( s [ i ] == '0' and count [ 0 ] > reqd and processed [ 0 ] >= reqd ) : NEW_LINE
if ( count [ 1 ] < reqd ) : NEW_LINE INDENT s [ i ] = '1' ; NEW_LINE count [ 1 ] += 1 ; NEW_LINE count [ 0 ] -= 1 ; NEW_LINE DEDENT
elif ( count [ 2 ] < reqd ) : NEW_LINE INDENT s [ i ] = '2' ; NEW_LINE count [ 2 ] += 1 ; NEW_LINE count [ 0 ] -= 1 ; NEW_LINE DEDENT
if ( s [ i ] == '1' and count [ 1 ] > reqd ) : NEW_LINE INDENT if ( count [ 0 ] < reqd ) : NEW_LINE INDENT s [ i ] = '0' ; NEW_LINE count [ 0 ] += 1 ; NEW_LINE count [ 1 ] -= 1 ; NEW_LINE DEDENT elif ( count [ 2 ] < reqd and processed [ 1 ] >= reqd ) : NEW_LINE INDENT s [ i ] = '2' ; NEW_LINE count [ 2 ] += 1 ; NEW_LINE count [ 1 ] -= 1 ; NEW_LINE DEDENT DEDENT
if ( s [ i ] == '2' and count [ 2 ] > reqd ) : NEW_LINE INDENT if ( count [ 0 ] < reqd ) : NEW_LINE INDENT s [ i ] = '0' ; NEW_LINE count [ 0 ] += 1 ; NEW_LINE count [ 2 ] -= 1 ; NEW_LINE DEDENT elif ( count [ 1 ] < reqd ) : NEW_LINE INDENT s [ i ] = '1' ; NEW_LINE count [ 1 ] += 1 ; NEW_LINE count [ 2 ] -= 1 ; NEW_LINE DEDENT DEDENT
processed [ ord ( s [ i ] ) - ord ( '0' ) ] += 1 ; NEW_LINE return ' ' . join ( s ) ; NEW_LINE
s = "011200" ; NEW_LINE print ( formStringMinOperations ( s ) ) ; NEW_LINE
def findMinimumAdjacentSwaps ( arr , N ) : NEW_LINE
visited = [ False ] * ( N + 1 ) NEW_LINE minimumSwaps = 0 NEW_LINE for i in range ( 2 * N ) : NEW_LINE
if ( visited [ arr [ i ] ] == False ) : NEW_LINE INDENT visited [ arr [ i ] ] = True NEW_LINE DEDENT
count = 0 NEW_LINE for j in range ( i + 1 , 2 * N ) : NEW_LINE
if ( visited [ arr [ j ] ] == False ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT
elif ( arr [ i ] == arr [ j ] ) : NEW_LINE INDENT minimumSwaps += count NEW_LINE DEDENT return minimumSwaps NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 1 , 2 , 3 , 3 , 1 , 2 ] NEW_LINE N = len ( arr ) NEW_LINE N //= 2 NEW_LINE print ( findMinimumAdjacentSwaps ( arr , N ) ) NEW_LINE DEDENT
from collections import defaultdict NEW_LINE
def possibility ( m , length , s ) : NEW_LINE
countodd = 0 NEW_LINE for i in range ( 0 , length ) : NEW_LINE
if m [ int ( s [ i ] ) ] & 1 : NEW_LINE INDENT countodd += 1 NEW_LINE DEDENT
if countodd > 1 : NEW_LINE INDENT return False NEW_LINE DEDENT return True NEW_LINE
def largestPalindrome ( s ) : NEW_LINE
l = len ( s ) NEW_LINE
m = defaultdict ( lambda : 0 ) NEW_LINE for i in range ( 0 , l ) : NEW_LINE INDENT m [ int ( s [ i ] ) ] += 1 NEW_LINE DEDENT
if possibility ( m , l , s ) == False : NEW_LINE INDENT print ( " Palindrome ▁ cannot ▁ be ▁ formed " ) NEW_LINE return NEW_LINE DEDENT
largest = [ None ] * l NEW_LINE
front = 0 NEW_LINE
for i in range ( 9 , - 1 , - 1 ) : NEW_LINE
if m [ i ] & 1 : NEW_LINE
largest [ l // 2 ] = chr ( i + 48 ) NEW_LINE
m [ i ] -= 1 NEW_LINE
while m [ i ] > 0 : NEW_LINE INDENT largest [ front ] = chr ( i + 48 ) NEW_LINE largest [ l - front - 1 ] = chr ( i + 48 ) NEW_LINE m [ i ] -= 2 NEW_LINE front += 1 NEW_LINE DEDENT else : NEW_LINE
while m [ i ] > 0 : NEW_LINE
largest [ front ] = chr ( i + 48 ) NEW_LINE largest [ l - front - 1 ] = chr ( i + 48 ) NEW_LINE
m [ i ] -= 2 NEW_LINE
front += 1 NEW_LINE
for i in range ( 0 , l ) : NEW_LINE INDENT print ( largest [ i ] , end = " " ) NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT s = "313551" NEW_LINE largestPalindrome ( s ) NEW_LINE DEDENT
def swapCount ( s ) : NEW_LINE
' NEW_LINE INDENT pos = [ ] NEW_LINE for i in range ( len ( s ) ) : NEW_LINE INDENT if ( s [ i ] == ' [ ' ) : NEW_LINE INDENT pos . append ( i ) NEW_LINE DEDENT DEDENT DEDENT
count = 0 NEW_LINE
p = 0 NEW_LINE
sum = 0 NEW_LINE s = list ( s ) NEW_LINE for i in range ( len ( s ) ) : NEW_LINE
if ( s [ i ] == ' [ ' ) : NEW_LINE INDENT count += 1 NEW_LINE p += 1 NEW_LINE DEDENT elif ( s [ i ] == ' ] ' ) : NEW_LINE INDENT count -= 1 NEW_LINE DEDENT
if ( count < 0 ) : NEW_LINE
sum += pos [ p ] - i NEW_LINE s [ i ] , s [ pos [ p ] ] = ( s [ pos [ p ] ] , s [ i ] ) NEW_LINE p += 1 NEW_LINE
count = 1 NEW_LINE return sum NEW_LINE
s = " [ ] ] [ ] [ " NEW_LINE print ( swapCount ( s ) ) NEW_LINE s = " [ [ ] [ ] ] " NEW_LINE print ( swapCount ( s ) ) NEW_LINE
def minimumCostOfBreaking ( X , Y , m , n ) : NEW_LINE INDENT res = 0 NEW_LINE DEDENT
X . sort ( reverse = True ) NEW_LINE
Y . sort ( reverse = True ) NEW_LINE
hzntl = 1 ; vert = 1 NEW_LINE
i = 0 ; j = 0 NEW_LINE while ( i < m and j < n ) : NEW_LINE INDENT if ( X [ i ] > Y [ j ] ) : NEW_LINE INDENT res += X [ i ] * vert NEW_LINE DEDENT DEDENT
hzntl += 1 NEW_LINE i += 1 NEW_LINE else : NEW_LINE res += Y [ j ] * hzntl NEW_LINE
vert += 1 NEW_LINE j += 1 NEW_LINE
total = 0 NEW_LINE while ( i < m ) : NEW_LINE INDENT total += X [ i ] NEW_LINE i += 1 NEW_LINE DEDENT res += total * vert NEW_LINE
total = 0 NEW_LINE while ( j < n ) : NEW_LINE INDENT total += Y [ j ] NEW_LINE j += 1 NEW_LINE DEDENT res += total * hzntl NEW_LINE return res NEW_LINE
m = 6 ; n = 4 NEW_LINE X = [ 2 , 1 , 3 , 1 , 4 ] NEW_LINE Y = [ 4 , 1 , 2 ] NEW_LINE print ( minimumCostOfBreaking ( X , Y , m - 1 , n - 1 ) ) NEW_LINE
def getMin ( x , y , z ) : NEW_LINE INDENT return min ( min ( x , y ) , z ) NEW_LINE DEDENT
def editDistance ( str1 , str2 , m , n ) : NEW_LINE
dp = [ [ 0 for i in range ( n + 1 ) ] for j in range ( m + 1 ) ] NEW_LINE
for i in range ( 0 , m + 1 ) : NEW_LINE INDENT for j in range ( 0 , n + 1 ) : NEW_LINE DEDENT
if ( i == 0 ) : NEW_LINE
dp [ i ] [ j ] = j NEW_LINE
elif ( j == 0 ) : NEW_LINE
dp [ i ] [ j ] = i NEW_LINE
elif ( str1 [ i - 1 ] == str2 [ j - 1 ] ) : NEW_LINE INDENT dp [ i ] [ j ] = dp [ i - 1 ] [ j - 1 ] NEW_LINE DEDENT
else : NEW_LINE
dp [ i ] [ j ] = 1 + getMin ( dp [ i ] [ j - 1 ] , dp [ i - 1 ] [ j ] , dp [ i - 1 ] [ j - 1 ] ) NEW_LINE
return dp [ m ] [ n ] NEW_LINE
def minimumSteps ( S , N ) : NEW_LINE
ans = 10 ** 10 NEW_LINE
for i in range ( 1 , N ) : NEW_LINE INDENT S1 = S [ : i ] NEW_LINE S2 = S [ i : ] NEW_LINE DEDENT
count = editDistance ( S1 , S2 , len ( S1 ) , len ( S2 ) ) NEW_LINE
ans = min ( ans , count ) NEW_LINE
print ( ans ) NEW_LINE
S = " aabb " NEW_LINE N = len ( S ) NEW_LINE minimumSteps ( S , N ) NEW_LINE
def minimumOperations ( N ) : NEW_LINE
dp = [ 0 for i in range ( N + 1 ) ] NEW_LINE
for i in range ( N + 1 ) : NEW_LINE INDENT dp [ i ] = 1000000000 NEW_LINE DEDENT
dp [ 2 ] = 0 NEW_LINE
for i in range ( 2 , N + 1 , 1 ) : NEW_LINE
if ( dp [ i ] == 1000000000 ) : NEW_LINE INDENT continue NEW_LINE DEDENT
if ( i * 5 <= N ) : NEW_LINE INDENT dp [ i * 5 ] = min ( dp [ i * 5 ] , dp [ i ] + 1 ) NEW_LINE DEDENT
if ( i + 3 <= N ) : NEW_LINE INDENT dp [ i + 3 ] = min ( dp [ i + 3 ] , dp [ i ] + 1 ) NEW_LINE DEDENT
if ( dp [ N ] == 1000000000 ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
return dp [ N ] NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 25 NEW_LINE print ( minimumOperations ( N ) ) NEW_LINE DEDENT
def MaxProfit ( arr , n , transactionFee ) : NEW_LINE INDENT buy = - arr [ 0 ] NEW_LINE sell = 0 NEW_LINE DEDENT
for i in range ( 1 , n , 1 ) : NEW_LINE INDENT temp = buy NEW_LINE DEDENT
buy = max ( buy , sell - arr [ i ] ) NEW_LINE sell = max ( sell , temp + arr [ i ] - transactionFee ) NEW_LINE
return max ( sell , buy ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
arr = [ 6 , 1 , 7 , 2 , 8 , 4 ] NEW_LINE n = len ( arr ) NEW_LINE transactionFee = 2 NEW_LINE
print ( MaxProfit ( arr , n , transactionFee ) ) NEW_LINE
start = [ [ 0 for i in range ( 3 ) ] for j in range ( 3 ) ] NEW_LINE
ending = [ [ 0 for i in range ( 3 ) ] for j in range ( 3 ) ] NEW_LINE
def calculateStart ( n , m ) : NEW_LINE
for i in range ( 1 , m , 1 ) : NEW_LINE INDENT start [ 0 ] [ i ] += start [ 0 ] [ i - 1 ] NEW_LINE DEDENT
for i in range ( 1 , n , 1 ) : NEW_LINE INDENT start [ i ] [ 0 ] += start [ i - 1 ] [ 0 ] NEW_LINE DEDENT
for i in range ( 1 , n , 1 ) : NEW_LINE INDENT for j in range ( 1 , m , 1 ) : NEW_LINE DEDENT
start [ i ] [ j ] += max ( start [ i - 1 ] [ j ] , start [ i ] [ j - 1 ] ) NEW_LINE
def calculateEnd ( n , m ) : NEW_LINE
i = n - 2 NEW_LINE while ( i >= 0 ) : NEW_LINE INDENT ending [ i ] [ m - 1 ] += ending [ i + 1 ] [ m - 1 ] NEW_LINE i -= 1 NEW_LINE DEDENT
i = m - 2 NEW_LINE while ( i >= 0 ) : NEW_LINE INDENT ending [ n - 1 ] [ i ] += ending [ n - 1 ] [ i + 1 ] NEW_LINE i -= 1 NEW_LINE DEDENT
i = n - 2 NEW_LINE while ( i >= 0 ) : NEW_LINE INDENT j = m - 2 NEW_LINE while ( j >= 0 ) : NEW_LINE DEDENT
ending [ i ] [ j ] += max ( ending [ i + 1 ] [ j ] , ending [ i ] [ j + 1 ] ) NEW_LINE j -= 1 NEW_LINE i -= 1 NEW_LINE
def maximumPathSum ( mat , n , m , q , coordinates ) : NEW_LINE
for i in range ( n ) : NEW_LINE INDENT for j in range ( m ) : NEW_LINE INDENT start [ i ] [ j ] = mat [ i ] [ j ] NEW_LINE ending [ i ] [ j ] = mat [ i ] [ j ] NEW_LINE DEDENT DEDENT
calculateStart ( n , m ) NEW_LINE
calculateEnd ( n , m ) NEW_LINE
ans = 0 NEW_LINE
for i in range ( q ) : NEW_LINE INDENT X = coordinates [ i ] [ 0 ] - 1 NEW_LINE Y = coordinates [ i ] [ 1 ] - 1 NEW_LINE DEDENT
ans = max ( ans , start [ X ] [ Y ] + ending [ X ] [ Y ] - mat [ X ] [ Y ] ) NEW_LINE
print ( ans ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT mat = [ [ 1 , 2 , 3 ] , [ 4 , 5 , 6 ] , [ 7 , 8 , 9 ] ] NEW_LINE N = 3 NEW_LINE M = 3 NEW_LINE Q = 2 NEW_LINE coordinates = [ [ 1 , 2 ] , [ 2 , 2 ] ] NEW_LINE maximumPathSum ( mat , N , M , Q , coordinates ) NEW_LINE DEDENT
def MaxSubsetlength ( arr , A , B ) : NEW_LINE
dp = [ [ 0 for i in range ( B + 1 ) ] for i in range ( A + 1 ) ] NEW_LINE
for str in arr : NEW_LINE
zeros = str . count ( '0' ) NEW_LINE ones = str . count ( '1' ) NEW_LINE
for i in range ( A , zeros - 1 , - 1 ) : NEW_LINE
for j in range ( B , ones - 1 , - 1 ) : NEW_LINE
dp [ i ] [ j ] = max ( dp [ i ] [ j ] , dp [ i - zeros ] [ j - ones ] + 1 ) NEW_LINE
return dp [ A ] [ B ] NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ "1" , "0" , "0001" , "10" , "111001" ] NEW_LINE A , B = 5 , 3 NEW_LINE print ( MaxSubsetlength ( arr , A , B ) ) NEW_LINE DEDENT
def numOfWays ( a , n , i = 0 , blue = [ ] ) : NEW_LINE
if i == n : NEW_LINE INDENT return 1 NEW_LINE DEDENT
count = 0 NEW_LINE
for j in range ( n ) : NEW_LINE
if mat [ i ] [ j ] == 1 and j not in blue : NEW_LINE INDENT count += numOfWays ( mat , n , i + 1 , blue + [ j ] ) NEW_LINE DEDENT return count NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 3 NEW_LINE mat = [ [ 0 , 1 , 1 ] , [ 1 , 0 , 1 ] , [ 1 , 1 , 1 ] ] NEW_LINE print ( numOfWays ( mat , n ) ) NEW_LINE DEDENT
def minCost ( arr , n ) : NEW_LINE
if ( n < 3 ) : NEW_LINE INDENT print ( arr [ 0 ] ) NEW_LINE return NEW_LINE DEDENT
dp = [ 0 ] * n NEW_LINE
dp [ 0 ] = arr [ 0 ] NEW_LINE dp [ 1 ] = dp [ 0 ] + arr [ 1 ] + arr [ 2 ] NEW_LINE
for i in range ( 2 , n - 1 ) : NEW_LINE INDENT dp [ i ] = min ( dp [ i - 2 ] + arr [ i ] , dp [ i - 1 ] + arr [ i ] + arr [ i + 1 ] ) NEW_LINE DEDENT
dp [ n - 1 ] = min ( dp [ n - 2 ] , dp [ n - 3 ] + arr [ n - 1 ] ) NEW_LINE
print ( dp [ n - 1 ] ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 9 , 4 , 6 , 8 , 5 ] NEW_LINE N = len ( arr ) NEW_LINE minCost ( arr , N ) NEW_LINE DEDENT
M = 1000000007 NEW_LINE
def power ( X , Y ) : NEW_LINE
res = 1 NEW_LINE
X = X % M NEW_LINE
if ( X == 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
while ( Y > 0 ) : NEW_LINE
if ( Y & 1 ) : NEW_LINE
res = ( res * X ) % M NEW_LINE
Y = Y >> 1 NEW_LINE
X = ( X * X ) % M NEW_LINE return res NEW_LINE
def findValue ( n ) : NEW_LINE
X = 0 NEW_LINE
pow_10 = 1 NEW_LINE
while ( n ) : NEW_LINE
if ( n & 1 ) : NEW_LINE
X += pow_10 NEW_LINE
pow_10 *= 10 NEW_LINE
n //= 2 NEW_LINE
X = ( X * 2 ) % M NEW_LINE
res = power ( 2 , X ) NEW_LINE return res NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 2 NEW_LINE print ( findValue ( n ) ) NEW_LINE DEDENT
M = 1000000007 ; NEW_LINE
def power ( X , Y ) : NEW_LINE
res = 1 ; NEW_LINE
X = X % M ; NEW_LINE
if ( X == 0 ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT
while ( Y > 0 ) : NEW_LINE
if ( Y % 2 == 1 ) : NEW_LINE
res = ( res * X ) % M ; NEW_LINE
Y = Y >> 1 ; NEW_LINE
X = ( X * X ) % M ; NEW_LINE return res ; NEW_LINE
def findValue ( N ) : NEW_LINE
dp = [ 0 ] * ( N + 1 ) ; NEW_LINE
dp [ 1 ] = 2 ; NEW_LINE dp [ 2 ] = 1024 ; NEW_LINE
for i in range ( 3 , N + 1 ) : NEW_LINE
y = ( i & ( - i ) ) ; NEW_LINE
x = i - y ; NEW_LINE
if ( x == 0 ) : NEW_LINE
dp [ i ] = power ( dp [ i // 2 ] , 10 ) ; NEW_LINE else : NEW_LINE
dp [ i ] = ( dp [ x ] * dp [ y ] ) % M ; NEW_LINE return ( dp [ N ] * dp [ N ] ) % M ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 150 ; NEW_LINE print ( findValue ( n ) ) ; NEW_LINE DEDENT
def findWays ( N ) : NEW_LINE
if ( N == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT
cnt = 0 NEW_LINE
for i in range ( 1 , 7 ) : NEW_LINE INDENT if ( N - i >= 0 ) : NEW_LINE INDENT cnt = cnt + findWays ( N - i ) NEW_LINE DEDENT DEDENT
return cnt NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 4 NEW_LINE DEDENT
print ( findWays ( N ) ) NEW_LINE
def checkEqualSumUtil ( arr , N , sm1 , sm2 , sm3 , j ) : NEW_LINE
if j == N : NEW_LINE INDENT if sm1 == sm2 and sm2 == sm3 : NEW_LINE INDENT return 1 NEW_LINE DEDENT else : NEW_LINE INDENT return 0 NEW_LINE DEDENT DEDENT else : NEW_LINE
l = checkEqualSumUtil ( arr , N , sm1 + arr [ j ] , sm2 , sm3 , j + 1 ) NEW_LINE
m = checkEqualSumUtil ( arr , N , sm1 , sm2 + arr [ j ] , sm3 , j + 1 ) NEW_LINE
r = checkEqualSumUtil ( arr , N , sm1 , sm2 , sm3 + arr [ j ] , j + 1 ) NEW_LINE
return max ( l , m , r ) NEW_LINE
def checkEqualSum ( arr , N ) : NEW_LINE
sum1 = sum2 = sum3 = 0 NEW_LINE
if checkEqualSumUtil ( arr , N , sum1 , sum2 , sum3 , 0 ) == 1 : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
arr = [ 17 , 34 , 59 , 23 , 17 , 67 , 57 , 2 , 18 , 59 , 1 ] NEW_LINE N = len ( arr ) NEW_LINE
checkEqualSum ( arr , N ) NEW_LINE
dp = { } NEW_LINE
def checkEqualSumUtil ( arr , N , sm1 , sm2 , sm3 , j ) : NEW_LINE INDENT s = str ( sm1 ) + " _ " + str ( sm2 ) + str ( j ) NEW_LINE DEDENT
if j == N : NEW_LINE INDENT if sm1 == sm2 and sm2 == sm3 : NEW_LINE INDENT return 1 NEW_LINE DEDENT else : NEW_LINE INDENT return 0 NEW_LINE DEDENT DEDENT
if s in dp : NEW_LINE INDENT return dp [ s ] NEW_LINE DEDENT
l = checkEqualSumUtil ( arr , N , sm1 + arr [ j ] , sm2 , sm3 , j + 1 ) NEW_LINE
m = checkEqualSumUtil ( arr , N , sm1 , sm2 + arr [ j ] , sm3 , j + 1 ) NEW_LINE
r = checkEqualSumUtil ( arr , N , sm1 , sm2 , sm3 + arr [ j ] , j + 1 ) NEW_LINE
dp [ s ] = max ( l , m , r ) NEW_LINE return dp [ s ] NEW_LINE
def checkEqualSum ( arr , N ) : NEW_LINE
sum1 = sum2 = sum3 = 0 NEW_LINE
if checkEqualSumUtil ( arr , N , sum1 , sum2 , sum3 , 0 ) == 1 : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
arr = [ 17 , 34 , 59 , 23 , 17 , 67 , 57 , 2 , 18 , 59 , 1 ] NEW_LINE N = len ( arr ) NEW_LINE
checkEqualSum ( arr , N ) NEW_LINE
def precompute ( nextpos , arr , N ) : NEW_LINE
nextpos [ N - 1 ] = N NEW_LINE for i in range ( N - 2 , - 1 , - 1 ) : NEW_LINE
if arr [ i ] == arr [ i + 1 ] : NEW_LINE INDENT nextpos [ i ] = nextpos [ i + 1 ] NEW_LINE DEDENT else : NEW_LINE INDENT nextpos [ i ] = i + 1 NEW_LINE DEDENT
def findIndex ( query , arr , N , Q ) : NEW_LINE
nextpos = [ 0 ] * N NEW_LINE precompute ( nextpos , arr , N ) NEW_LINE for i in range ( Q ) : NEW_LINE INDENT l = query [ i ] [ 0 ] NEW_LINE r = query [ i ] [ 1 ] NEW_LINE x = query [ i ] [ 2 ] NEW_LINE ans = - 1 NEW_LINE DEDENT
if arr [ l ] != x : NEW_LINE INDENT ans = l NEW_LINE DEDENT
else : NEW_LINE
d = nextpos [ l ] NEW_LINE
if d <= r : NEW_LINE INDENT ans = d NEW_LINE DEDENT print ( ans ) NEW_LINE
N = 6 NEW_LINE Q = 3 NEW_LINE arr = [ 1 , 2 , 1 , 1 , 3 , 5 ] NEW_LINE query = [ [ 0 , 3 , 1 ] , [ 1 , 5 , 2 ] , [ 2 , 3 , 1 ] ] NEW_LINE findIndex ( query , arr , N , Q ) NEW_LINE
mod = 1000000007 NEW_LINE
def countWays ( s , t , k ) : NEW_LINE
n = len ( s ) NEW_LINE
a = 0 NEW_LINE b = 0 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT p = s [ i : n - i + 1 ] + s [ : i + 1 ] NEW_LINE DEDENT
if ( p == t ) : NEW_LINE INDENT a += 1 NEW_LINE DEDENT else : NEW_LINE INDENT b += 1 NEW_LINE DEDENT
dp1 = [ 0 ] * ( k + 1 ) NEW_LINE dp2 = [ 0 ] * ( k + 1 ) NEW_LINE if ( s == t ) : NEW_LINE INDENT dp1 [ 0 ] = 1 NEW_LINE dp2 [ 0 ] = 0 NEW_LINE DEDENT else : NEW_LINE INDENT dp1 [ 0 ] = 0 NEW_LINE dp2 [ 0 ] = 1 NEW_LINE DEDENT
for i in range ( 1 , k + 1 ) : NEW_LINE INDENT dp1 [ i ] = ( ( dp1 [ i - 1 ] * ( a - 1 ) ) % mod + ( dp2 [ i - 1 ] * a ) % mod ) % mod NEW_LINE dp2 [ i ] = ( ( dp1 [ i - 1 ] * ( b ) ) % mod + ( dp2 [ i - 1 ] * ( b - 1 ) ) % mod ) % mod NEW_LINE DEDENT
return ( dp1 [ k ] ) NEW_LINE
S = ' ab ' NEW_LINE T = ' ab ' NEW_LINE
K = 2 NEW_LINE
print ( countWays ( S , T , K ) ) NEW_LINE
def minOperation ( k ) : NEW_LINE
dp = [ 0 ] * ( k + 1 ) NEW_LINE for i in range ( 1 , k + 1 ) : NEW_LINE INDENT dp [ i ] = dp [ i - 1 ] + 1 NEW_LINE DEDENT
if ( i % 2 == 0 ) : NEW_LINE INDENT dp [ i ] = min ( dp [ i ] , dp [ i // 2 ] + 1 ) NEW_LINE DEDENT return dp [ k ] NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT k = 12 NEW_LINE print ( minOperation ( k ) ) NEW_LINE DEDENT
def maxSum ( p0 , p1 , a , pos , n ) : NEW_LINE INDENT if ( pos == n ) : NEW_LINE INDENT if ( p0 == p1 ) : NEW_LINE INDENT return p0 ; NEW_LINE DEDENT else : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT DEDENT DEDENT
ans = maxSum ( p0 , p1 , a , pos + 1 , n ) ; NEW_LINE
ans = max ( ans , maxSum ( p0 + a [ pos ] , p1 , a , pos + 1 , n ) ) ; NEW_LINE
ans = max ( ans , maxSum ( p0 , p1 + a [ pos ] , a , pos + 1 , n ) ) ; NEW_LINE return ans ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
n = 4 ; NEW_LINE a = [ 1 , 2 , 3 , 6 ] ; NEW_LINE print ( maxSum ( 0 , 0 , a , 0 , n ) ) ; NEW_LINE
import numpy as np NEW_LINE import sys NEW_LINE INT_MIN = - ( sys . maxsize - 1 ) NEW_LINE
def maxSum ( a , n ) : NEW_LINE
sum = 0 ; NEW_LINE for i in range ( n ) : NEW_LINE INDENT sum += a [ i ] ; NEW_LINE DEDENT limit = 2 * sum + 1 ; NEW_LINE
dp = np . zeros ( ( n + 1 , limit ) ) ; NEW_LINE
for i in range ( n + 1 ) : NEW_LINE INDENT for j in range ( limit ) : NEW_LINE INDENT dp [ i ] [ j ] = INT_MIN ; NEW_LINE DEDENT DEDENT
dp [ 0 ] [ sum ] = 0 ; NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE INDENT for j in range ( limit ) : NEW_LINE DEDENT
if ( ( j - a [ i - 1 ] ) >= 0 and dp [ i - 1 ] [ j - a [ i - 1 ] ] != INT_MIN ) : NEW_LINE INDENT dp [ i ] [ j ] = max ( dp [ i ] [ j ] , dp [ i - 1 ] [ j - a [ i - 1 ] ] + a [ i - 1 ] ) ; NEW_LINE DEDENT
if ( ( j + a [ i - 1 ] ) < limit and dp [ i - 1 ] [ j + a [ i - 1 ] ] != INT_MIN ) : NEW_LINE INDENT dp [ i ] [ j ] = max ( dp [ i ] [ j ] , dp [ i - 1 ] [ j + a [ i - 1 ] ] ) ; NEW_LINE DEDENT
if ( dp [ i - 1 ] [ j ] != INT_MIN ) : NEW_LINE INDENT dp [ i ] [ j ] = max ( dp [ i ] [ j ] , dp [ i - 1 ] [ j ] ) ; NEW_LINE DEDENT return dp [ n ] [ sum ] ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 4 ; NEW_LINE a = [ 1 , 2 , 3 , 6 ] ; NEW_LINE print ( maxSum ( a , n ) ) ; NEW_LINE DEDENT
fib = [ 0 ] * 100005 ; NEW_LINE
def computeFibonacci ( ) : NEW_LINE INDENT fib [ 0 ] = 1 ; NEW_LINE fib [ 1 ] = 1 ; NEW_LINE for i in range ( 2 , 100005 ) : NEW_LINE INDENT fib [ i ] = fib [ i - 1 ] + fib [ i - 2 ] ; NEW_LINE DEDENT DEDENT
def countString ( string ) : NEW_LINE
ans = 1 ; NEW_LINE cnt = 1 ; NEW_LINE for i in range ( 1 , len ( string ) ) : NEW_LINE
if ( string [ i ] == string [ i - 1 ] ) : NEW_LINE INDENT cnt += 1 ; NEW_LINE DEDENT
else : NEW_LINE INDENT ans = ans * fib [ cnt ] ; NEW_LINE cnt = 1 ; NEW_LINE DEDENT
ans = ans * fib [ cnt ] ; NEW_LINE
return ans ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT string = " abdllldefkkkk " ; NEW_LINE DEDENT
computeFibonacci ( ) ; NEW_LINE
print ( countString ( string ) ) ; NEW_LINE
MAX = 100001 NEW_LINE
def printGolombSequence ( N ) : NEW_LINE
arr = [ 0 ] * MAX NEW_LINE
cnt = 0 NEW_LINE
arr [ 0 ] = 0 NEW_LINE arr [ 1 ] = 1 NEW_LINE
M = dict ( ) NEW_LINE
M [ 2 ] = 2 NEW_LINE
for i in range ( 2 , N + 1 ) : NEW_LINE
if ( cnt == 0 ) : NEW_LINE INDENT arr [ i ] = 1 + arr [ i - 1 ] NEW_LINE cnt = M [ arr [ i ] ] NEW_LINE cnt -= 1 NEW_LINE DEDENT
else : NEW_LINE INDENT arr [ i ] = arr [ i - 1 ] NEW_LINE cnt -= 1 NEW_LINE DEDENT
M [ i ] = arr [ i ] NEW_LINE
for i in range ( 1 , N + 1 ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT
N = 11 NEW_LINE printGolombSequence ( N ) NEW_LINE
def number_of_ways ( n ) : NEW_LINE
includes_3 = [ 0 ] * ( n + 1 ) NEW_LINE
not_includes_3 = [ 0 ] * ( n + 1 ) NEW_LINE
includes_3 [ 3 ] = 1 NEW_LINE not_includes_3 [ 1 ] = 1 NEW_LINE not_includes_3 [ 2 ] = 2 NEW_LINE not_includes_3 [ 3 ] = 3 NEW_LINE
for i in range ( 4 , n + 1 ) : NEW_LINE INDENT includes_3 [ i ] = includes_3 [ i - 1 ] + includes_3 [ i - 2 ] + not_includes_3 [ i - 3 ] NEW_LINE not_includes_3 [ i ] = not_includes_3 [ i - 1 ] + not_includes_3 [ i - 2 ] NEW_LINE DEDENT return includes_3 [ n ] NEW_LINE
n = 7 NEW_LINE print ( number_of_ways ( n ) ) NEW_LINE
from math import ceil , sqrt NEW_LINE MAX = 100000 NEW_LINE
divisors = [ 0 ] * MAX NEW_LINE
def generateDivisors ( n ) : NEW_LINE INDENT for i in range ( 1 , ceil ( sqrt ( n ) ) + 1 ) : NEW_LINE INDENT if ( n % i == 0 ) : NEW_LINE INDENT if ( n // i == i ) : NEW_LINE INDENT divisors [ i ] += 1 NEW_LINE DEDENT else : NEW_LINE INDENT divisors [ i ] += 1 NEW_LINE divisors [ n // i ] += 1 NEW_LINE DEDENT DEDENT DEDENT DEDENT
def findMaxMultiples ( arr , n ) : NEW_LINE
ans = 0 NEW_LINE for i in range ( n ) : NEW_LINE
ans = max ( divisors [ arr [ i ] ] , ans ) NEW_LINE
generateDivisors ( arr [ i ] ) NEW_LINE return ans NEW_LINE
arr = [ 8 , 1 , 28 , 4 , 2 , 6 , 7 ] NEW_LINE n = len ( arr ) NEW_LINE print ( findMaxMultiples ( arr , n ) ) NEW_LINE
n = 3 NEW_LINE maxV = 20 NEW_LINE
dp = [ [ [ 0 for i in range ( maxV ) ] for i in range ( n ) ] for i in range ( n ) ] NEW_LINE
v = [ [ [ 0 for i in range ( maxV ) ] for i in range ( n ) ] for i in range ( n ) ] NEW_LINE
def countWays ( i , j , x , arr ) : NEW_LINE
if ( i == n or j == n ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT x = ( x & arr [ i ] [ j ] ) NEW_LINE if ( x == 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT if ( i == n - 1 and j == n - 1 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT
if ( v [ i ] [ j ] [ x ] ) : NEW_LINE INDENT return dp [ i ] [ j ] [ x ] NEW_LINE DEDENT v [ i ] [ j ] [ x ] = 1 NEW_LINE
dp [ i ] [ j ] [ x ] = countWays ( i + 1 , j , x , arr ) + countWays ( i , j + 1 , x , arr ) ; NEW_LINE return dp [ i ] [ j ] [ x ] NEW_LINE
arr = [ [ 1 , 2 , 1 ] , [ 1 , 1 , 0 ] , [ 2 , 1 , 1 ] ] NEW_LINE print ( countWays ( 0 , 0 , arr [ 0 ] [ 0 ] , arr ) ) NEW_LINE
def FindMaximumSum ( ind , kon , a , b , c , n , dp ) : NEW_LINE
if ind == n : NEW_LINE INDENT return 0 NEW_LINE DEDENT
if dp [ ind ] [ kon ] != - 1 : NEW_LINE INDENT return dp [ ind ] [ kon ] NEW_LINE DEDENT ans = - 10 ** 9 + 5 NEW_LINE
if kon == 0 : NEW_LINE INDENT ans = max ( ans , b [ ind ] + FindMaximumSum ( ind + 1 , 1 , a , b , c , n , dp ) ) NEW_LINE ans = max ( ans , c [ ind ] + FindMaximumSum ( ind + 1 , 2 , a , b , c , n , dp ) ) NEW_LINE DEDENT
elif kon == 1 : NEW_LINE INDENT ans = max ( ans , a [ ind ] + FindMaximumSum ( ind + 1 , 0 , a , b , c , n , dp ) ) NEW_LINE ans = max ( ans , c [ ind ] + FindMaximumSum ( ind + 1 , 2 , a , b , c , n , dp ) ) NEW_LINE DEDENT
elif kon == 2 : NEW_LINE INDENT ans = max ( ans , a [ ind ] + FindMaximumSum ( ind + 1 , 1 , a , b , c , n , dp ) ) NEW_LINE ans = max ( ans , b [ ind ] + FindMaximumSum ( ind + 1 , 0 , a , b , c , n , dp ) ) NEW_LINE DEDENT dp [ ind ] [ kon ] = ans NEW_LINE return ans NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT N = 3 NEW_LINE a = [ 6 , 8 , 2 , 7 , 4 , 2 , 7 ] NEW_LINE b = [ 7 , 8 , 5 , 8 , 6 , 3 , 5 ] NEW_LINE c = [ 8 , 3 , 2 , 6 , 8 , 4 , 1 ] NEW_LINE n = len ( a ) NEW_LINE dp = [ [ - 1 for i in range ( N ) ] for j in range ( n ) ] NEW_LINE DEDENT
x = FindMaximumSum ( 0 , 0 , a , b , c , n , dp ) NEW_LINE
y = FindMaximumSum ( 0 , 1 , a , b , c , n , dp ) NEW_LINE
z = FindMaximumSum ( 0 , 2 , a , b , c , n , dp ) NEW_LINE
print ( max ( x , y , z ) ) NEW_LINE
mod = 1000000007 ; NEW_LINE
def noOfBinaryStrings ( N , k ) : NEW_LINE INDENT dp = [ 0 ] * 100002 ; NEW_LINE for i in range ( 1 , K ) : NEW_LINE INDENT dp [ i ] = 1 ; NEW_LINE DEDENT dp [ k ] = 2 ; NEW_LINE for i in range ( k + 1 , N + 1 ) : NEW_LINE INDENT dp [ i ] = ( dp [ i - 1 ] + dp [ i - k ] ) % mod ; NEW_LINE DEDENT return dp [ N ] ; NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT N = 4 ; NEW_LINE K = 2 ; NEW_LINE print ( noOfBinaryStrings ( N , K ) ) ; NEW_LINE DEDENT
def findWays ( p ) : NEW_LINE
dp = [ 0 ] * ( p + 1 ) NEW_LINE dp [ 1 ] = 1 NEW_LINE dp [ 2 ] = 2 NEW_LINE
for i in range ( 3 , p + 1 ) : NEW_LINE INDENT dp [ i ] = ( dp [ i - 1 ] + ( i - 1 ) * dp [ i - 2 ] ) NEW_LINE DEDENT return dp [ p ] NEW_LINE
p = 3 NEW_LINE print ( findWays ( p ) ) NEW_LINE
def CountWays ( n ) : NEW_LINE
if n == 0 : NEW_LINE INDENT return 1 NEW_LINE DEDENT if n == 1 : NEW_LINE INDENT return 1 NEW_LINE DEDENT if n == 2 : NEW_LINE INDENT return 1 + 1 NEW_LINE DEDENT
return CountWays ( n - 1 ) + CountWays ( n - 3 ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 5 NEW_LINE print ( CountWays ( n ) ) NEW_LINE DEDENT
from math import sqrt NEW_LINE
def factors ( n ) : NEW_LINE
v = [ ] NEW_LINE v . append ( 1 ) NEW_LINE
for i in range ( 2 , int ( sqrt ( n ) ) + 1 , 1 ) : NEW_LINE
if ( n % i == 0 ) : NEW_LINE INDENT v . append ( i ) ; NEW_LINE DEDENT
if ( int ( n / i ) != i ) : NEW_LINE INDENT v . append ( int ( n / i ) ) NEW_LINE DEDENT
return v NEW_LINE
def checkAbundant ( n ) : NEW_LINE INDENT sum = 0 NEW_LINE DEDENT
v = factors ( n ) NEW_LINE
for i in range ( len ( v ) ) : NEW_LINE INDENT sum += v [ i ] NEW_LINE DEDENT
if ( sum > n ) : NEW_LINE INDENT return True NEW_LINE DEDENT else : NEW_LINE INDENT return False NEW_LINE DEDENT
def checkSemiPerfect ( n ) : NEW_LINE
v = factors ( n ) NEW_LINE
v . sort ( reverse = False ) NEW_LINE r = len ( v ) NEW_LINE
subset = [ [ 0 for i in range ( n + 1 ) ] for j in range ( r + 1 ) ] NEW_LINE
for i in range ( r + 1 ) : NEW_LINE INDENT subset [ i ] [ 0 ] = True NEW_LINE DEDENT
for i in range ( 1 , n + 1 ) : NEW_LINE INDENT subset [ 0 ] [ i ] = False NEW_LINE DEDENT
for i in range ( 1 , r + 1 ) : NEW_LINE INDENT for j in range ( 1 , n + 1 ) : NEW_LINE DEDENT
if ( j < v [ i - 1 ] ) : NEW_LINE INDENT subset [ i ] [ j ] = subset [ i - 1 ] [ j ] NEW_LINE DEDENT else : NEW_LINE INDENT subset [ i ] [ j ] = ( subset [ i - 1 ] [ j ] or subset [ i - 1 ] [ j - v [ i - 1 ] ] ) NEW_LINE DEDENT
if ( ( subset [ r ] [ n ] ) == 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT else : NEW_LINE INDENT return True NEW_LINE DEDENT
def checkweird ( n ) : NEW_LINE INDENT if ( checkAbundant ( n ) == True and checkSemiPerfect ( n ) == False ) : NEW_LINE INDENT return True NEW_LINE DEDENT else : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 70 NEW_LINE if ( checkweird ( n ) ) : NEW_LINE INDENT print ( " Weird ▁ Number " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Not ▁ Weird ▁ Number " ) NEW_LINE DEDENT DEDENT
def maxSubArraySumRepeated ( a , n , k ) : NEW_LINE INDENT max_so_far = - 2147483648 NEW_LINE max_ending_here = 0 NEW_LINE for i in range ( n * k ) : NEW_LINE DEDENT
max_ending_here = max_ending_here + a [ i % n ] NEW_LINE if ( max_so_far < max_ending_here ) : NEW_LINE INDENT max_so_far = max_ending_here NEW_LINE DEDENT if ( max_ending_here < 0 ) : NEW_LINE INDENT max_ending_here = 0 NEW_LINE DEDENT return max_so_far NEW_LINE
a = [ 10 , 20 , - 30 , - 1 ] NEW_LINE n = len ( a ) NEW_LINE k = 3 NEW_LINE print ( " Maximum ▁ contiguous ▁ sum ▁ is ▁ " , maxSubArraySumRepeated ( a , n , k ) ) NEW_LINE
def longOddEvenIncSeq ( arr , n ) : NEW_LINE
lioes = list ( ) NEW_LINE
maxLen = 0 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT lioes . append ( 1 ) NEW_LINE DEDENT
i = 1 NEW_LINE for i in range ( n ) : NEW_LINE INDENT for j in range ( i ) : NEW_LINE INDENT if ( arr [ i ] > arr [ j ] and ( arr [ i ] + arr [ j ] ) % 2 != 0 and lioes [ i ] < lioes [ j ] + 1 ) : NEW_LINE INDENT lioes [ i ] = lioes [ j ] + 1 NEW_LINE DEDENT DEDENT DEDENT
for i in range ( n ) : NEW_LINE INDENT if maxLen < lioes [ i ] : NEW_LINE INDENT maxLen = lioes [ i ] NEW_LINE DEDENT DEDENT
return maxLen NEW_LINE
arr = [ 1 , 12 , 2 , 22 , 5 , 30 , 31 , 14 , 17 , 11 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Longest ▁ Increasing ▁ Odd ▁ Even ▁ " + " Subsequence : ▁ " , longOddEvenIncSeq ( arr , n ) ) NEW_LINE
def isOperator ( op ) : NEW_LINE INDENT return ( op == ' + ' or op == ' * ' ) NEW_LINE DEDENT
def printMinAndMaxValueOfExp ( exp ) : NEW_LINE INDENT num = [ ] NEW_LINE opr = [ ] NEW_LINE tmp = " " NEW_LINE DEDENT
for i in range ( len ( exp ) ) : NEW_LINE INDENT if ( isOperator ( exp [ i ] ) ) : NEW_LINE INDENT opr . append ( exp [ i ] ) NEW_LINE num . append ( int ( tmp ) ) NEW_LINE tmp = " " NEW_LINE DEDENT else : NEW_LINE INDENT tmp += exp [ i ] NEW_LINE DEDENT DEDENT
num . append ( int ( tmp ) ) NEW_LINE llen = len ( num ) NEW_LINE minVal = [ [ 0 for i in range ( llen ) ] for i in range ( llen ) ] NEW_LINE maxVal = [ [ 0 for i in range ( llen ) ] for i in range ( llen ) ] NEW_LINE
for i in range ( llen ) : NEW_LINE INDENT for j in range ( llen ) : NEW_LINE INDENT minVal [ i ] [ j ] = 10 ** 9 NEW_LINE maxVal [ i ] [ j ] = 0 NEW_LINE DEDENT DEDENT
if ( i == j ) : NEW_LINE INDENT minVal [ i ] [ j ] = maxVal [ i ] [ j ] = num [ i ] NEW_LINE DEDENT
for L in range ( 2 , llen + 1 ) : NEW_LINE INDENT for i in range ( llen - L + 1 ) : NEW_LINE INDENT j = i + L - 1 NEW_LINE for k in range ( i , j ) : NEW_LINE INDENT minTmp = 0 NEW_LINE maxTmp = 0 NEW_LINE DEDENT DEDENT DEDENT
if ( opr [ k ] == ' + ' ) : NEW_LINE INDENT minTmp = minVal [ i ] [ k ] + minVal [ k + 1 ] [ j ] NEW_LINE maxTmp = maxVal [ i ] [ k ] + maxVal [ k + 1 ] [ j ] NEW_LINE DEDENT
elif ( opr [ k ] == ' * ' ) : NEW_LINE INDENT minTmp = minVal [ i ] [ k ] * minVal [ k + 1 ] [ j ] NEW_LINE maxTmp = maxVal [ i ] [ k ] * maxVal [ k + 1 ] [ j ] NEW_LINE DEDENT
if ( minTmp < minVal [ i ] [ j ] ) : NEW_LINE INDENT minVal [ i ] [ j ] = minTmp NEW_LINE DEDENT if ( maxTmp > maxVal [ i ] [ j ] ) : NEW_LINE INDENT maxVal [ i ] [ j ] = maxTmp NEW_LINE DEDENT
print ( " Minimum ▁ value ▁ : ▁ " , minVal [ 0 ] [ llen - 1 ] , " , ▁ \ ▁ Maximum ▁ value ▁ : ▁ " , maxVal [ 0 ] [ llen - 1 ] ) NEW_LINE
expression = "1 + 2*3 + 4*5" NEW_LINE printMinAndMaxValueOfExp ( expression ) NEW_LINE
import sys NEW_LINE
def MatrixChainOrder ( p , i , j ) : NEW_LINE INDENT if i == j : NEW_LINE INDENT return 0 NEW_LINE DEDENT _min = sys . maxsize NEW_LINE DEDENT
for k in range ( i , j ) : NEW_LINE INDENT count = ( MatrixChainOrder ( p , i , k ) + MatrixChainOrder ( p , k + 1 , j ) + p [ i - 1 ] * p [ k ] * p [ j ] ) NEW_LINE if count < _min : NEW_LINE INDENT _min = count NEW_LINE DEDENT DEDENT
return _min NEW_LINE
arr = [ 1 , 2 , 3 , 4 , 3 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Minimum ▁ number ▁ of ▁ multiplications ▁ is ▁ " , MatrixChainOrder ( arr , 1 , n - 1 ) ) NEW_LINE
import sys NEW_LINE dp = [ [ - 1 for i in range ( 100 ) ] for j in range ( 100 ) ] NEW_LINE
def matrixChainMemoised ( p , i , j ) : NEW_LINE INDENT if ( i == j ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT if ( dp [ i ] [ j ] != - 1 ) : NEW_LINE INDENT return dp [ i ] [ j ] NEW_LINE DEDENT dp [ i ] [ j ] = sys . maxsize NEW_LINE for k in range ( i , j ) : NEW_LINE INDENT dp [ i ] [ j ] = min ( dp [ i ] [ j ] , matrixChainMemoised ( p , i , k ) + matrixChainMemoised ( p , k + 1 , j ) + p [ i - 1 ] * p [ k ] * p [ j ] ) NEW_LINE DEDENT return dp [ i ] [ j ] NEW_LINE DEDENT def MatrixChainOrder ( p , n ) : NEW_LINE INDENT i = 1 NEW_LINE j = n - 1 NEW_LINE return matrixChainMemoised ( p , i , j ) NEW_LINE DEDENT
arr = [ 1 , 2 , 3 , 4 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Minimum ▁ number ▁ of ▁ multiplications ▁ is " , MatrixChainOrder ( arr , n ) ) NEW_LINE
def flipBitsOfAandB ( A , B ) : NEW_LINE
A = A ^ ( A & B ) NEW_LINE
B = B ^ ( A & B ) NEW_LINE
print ( A , B ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT A = 10 NEW_LINE B = 20 NEW_LINE flipBitsOfAandB ( A , B ) NEW_LINE DEDENT
def TotalHammingDistance ( n ) : NEW_LINE INDENT i = 1 NEW_LINE sum = 0 NEW_LINE while ( n // i > 0 ) : NEW_LINE INDENT sum = sum + n // i NEW_LINE i = i * 2 NEW_LINE DEDENT return sum NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 9 NEW_LINE print ( TotalHammingDistance ( N ) ) NEW_LINE DEDENT
import math NEW_LINE m = 1000000007 NEW_LINE
def solve ( n ) : NEW_LINE
s = 0 ; NEW_LINE l = 1 ; NEW_LINE while ( l < n + 1 ) : NEW_LINE
r = ( int ) ( n / math . floor ( n / l ) ) ; NEW_LINE x = ( ( ( ( r % m ) * ( ( r + 1 ) % m ) ) / 2 ) % m ) ; NEW_LINE y = ( ( ( ( l % m ) * ( ( l - 1 ) % m ) ) / 2 ) % m ) ; NEW_LINE p = ( int ) ( ( n / l ) % m ) ; NEW_LINE
s = ( ( s + ( ( ( x - y ) % m ) * p ) % m + m ) % m ) ; NEW_LINE s %= m ; NEW_LINE l = r + 1 ; NEW_LINE
print ( int ( ( s + m ) % m ) ) ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 12 ; NEW_LINE solve ( n ) ; NEW_LINE DEDENT
import math NEW_LINE
def min_time_to_cut ( N ) : NEW_LINE INDENT if ( N == 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT DEDENT
return int ( math . log2 ( N ) ) + 1 NEW_LINE
N = 100 NEW_LINE print ( min_time_to_cut ( N ) ) NEW_LINE
def findDistinctSums ( n ) : NEW_LINE
s = set ( ) NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE INDENT for j in range ( i , n + 1 ) : NEW_LINE DEDENT
s . add ( i + j ) NEW_LINE
return len ( s ) NEW_LINE
N = 3 NEW_LINE print ( findDistinctSums ( N ) ) NEW_LINE
def printPattern ( i , j , n ) : NEW_LINE
if ( j >= n ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT if ( i >= n ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT
if ( j == i or j == n - 1 - i ) : NEW_LINE
if ( i == n - 1 - j ) : NEW_LINE INDENT print ( " / " , end = " " ) NEW_LINE DEDENT
else : NEW_LINE INDENT print ( " \\ " , end = " " ) NEW_LINE DEDENT
' NEW_LINE INDENT else : NEW_LINE INDENT print ( " * " , end = " " ) NEW_LINE DEDENT DEDENT
if ( printPattern ( i , j + 1 , n ) == 1 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT print ( ) NEW_LINE
return printPattern ( i + 1 , 0 , n ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT N = 9 NEW_LINE DEDENT
printPattern ( 0 , 0 , N ) NEW_LINE
import sys ; NEW_LINE
def zArray ( arr ) : NEW_LINE INDENT n = len ( arr ) ; NEW_LINE z = [ 0 ] * n ; NEW_LINE r = 0 ; NEW_LINE l = 0 ; NEW_LINE DEDENT
for k in range ( 1 , n ) : NEW_LINE
if ( k > r ) : NEW_LINE INDENT r = l = k ; NEW_LINE while ( r < n and arr [ r ] == arr [ r - l ] ) : NEW_LINE INDENT r += 1 ; NEW_LINE DEDENT z [ k ] = r - l ; NEW_LINE r -= 1 ; NEW_LINE DEDENT
else : NEW_LINE INDENT k1 = k - l ; NEW_LINE if ( z [ k1 ] < r - k + 1 ) : NEW_LINE INDENT z [ k ] = z [ k1 ] ; NEW_LINE DEDENT else : NEW_LINE INDENT l = k ; NEW_LINE while ( r < n and arr [ r ] == arr [ r - l ] ) : NEW_LINE INDENT r += 1 ; NEW_LINE DEDENT z [ k ] = r - l ; NEW_LINE r -= 1 ; NEW_LINE DEDENT DEDENT return z ; NEW_LINE
def mergeArray ( A , B ) : NEW_LINE INDENT n = len ( A ) ; NEW_LINE m = len ( B ) ; NEW_LINE DEDENT
c = [ 0 ] * ( n + m + 1 ) ; NEW_LINE
for i in range ( m ) : NEW_LINE INDENT c [ i ] = B [ i ] ; NEW_LINE DEDENT
c [ m ] = sys . maxsize ; NEW_LINE
for i in range ( n ) : NEW_LINE INDENT c [ m + i + 1 ] = A [ i ] ; NEW_LINE DEDENT
z = zArray ( c ) ; NEW_LINE return z ; NEW_LINE
def findZArray ( A , B , n ) : NEW_LINE INDENT flag = 0 ; NEW_LINE z = mergeArray ( A , B ) ; NEW_LINE DEDENT
for i in range ( len ( z ) ) : NEW_LINE INDENT if ( z [ i ] == n ) : NEW_LINE INDENT print ( i - n - 1 , end = " ▁ " ) ; NEW_LINE flag = 1 ; NEW_LINE DEDENT DEDENT if ( flag == 0 ) : NEW_LINE INDENT print ( " Not ▁ Found " ) ; NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT A = [ 1 , 2 , 3 , 2 , 3 , 2 ] ; NEW_LINE B = [ 2 , 3 ] ; NEW_LINE n = len ( B ) ; NEW_LINE findZArray ( A , B , n ) ; NEW_LINE DEDENT
def getCount ( a , b ) : NEW_LINE
if ( len ( b ) % len ( a ) != 0 ) : NEW_LINE INDENT return - 1 ; NEW_LINE DEDENT count = int ( len ( b ) / len ( a ) ) NEW_LINE
a = a * count NEW_LINE if ( a == b ) : NEW_LINE INDENT return count NEW_LINE DEDENT return - 1 ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT a = ' geeks ' NEW_LINE b = ' geeksgeeks ' NEW_LINE print ( getCount ( a , b ) ) NEW_LINE DEDENT
from collections import defaultdict NEW_LINE
def check ( S1 , S2 ) : NEW_LINE
n1 = len ( S1 ) NEW_LINE n2 = len ( S2 ) NEW_LINE
mp = defaultdict ( lambda : 0 ) NEW_LINE
for i in range ( 0 , n1 ) : NEW_LINE INDENT mp [ S1 [ i ] ] += 1 NEW_LINE DEDENT
for i in range ( 0 , n2 ) : NEW_LINE
if mp [ S2 [ i ] ] : NEW_LINE INDENT mp [ S2 [ i ] ] -= 1 NEW_LINE DEDENT
elif ( mp [ chr ( ord ( S2 [ i ] ) - 1 ) ] and mp [ chr ( ord ( S2 [ i ] ) - 2 ) ] ) : NEW_LINE INDENT mp [ chr ( ord ( S2 [ i ] ) - 1 ) ] -= 1 NEW_LINE mp [ chr ( ord ( S2 [ i ] ) - 2 ) ] -= 1 NEW_LINE DEDENT else : NEW_LINE INDENT return False NEW_LINE DEDENT return True NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT S1 = " abbat " NEW_LINE S2 = " cat " NEW_LINE DEDENT
if check ( S1 , S2 ) : NEW_LINE INDENT print ( " YES " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NO " ) NEW_LINE DEDENT
def countPattern ( s ) : NEW_LINE INDENT length = len ( s ) NEW_LINE oneSeen = False NEW_LINE DEDENT
for i in range ( length ) : NEW_LINE
if ( s [ i ] == '1' and oneSeen ) : NEW_LINE INDENT if ( s [ i - 1 ] == '0' ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT DEDENT
if ( s [ i ] == '1' and oneSeen == 0 ) : NEW_LINE INDENT oneSeen = True NEW_LINE DEDENT
if ( s [ i ] != '0' and s [ i ] != '1' ) : NEW_LINE INDENT oneSeen = False NEW_LINE DEDENT return count NEW_LINE
s = "100001abc101" NEW_LINE print countPattern ( s ) NEW_LINE
def checkIfPossible ( N , arr , T ) : NEW_LINE
freqS = [ 0 ] * 256 NEW_LINE
freqT = [ 0 ] * 256 NEW_LINE
for ch in T : NEW_LINE INDENT freqT [ ord ( ch ) - ord ( ' a ' ) ] += 1 NEW_LINE DEDENT
for i in range ( N ) : NEW_LINE
for ch in arr [ i ] : NEW_LINE INDENT freqS [ ord ( ch ) - ord ( ' a ' ) ] += 1 NEW_LINE DEDENT for i in range ( 256 ) : NEW_LINE
if ( freqT [ i ] == 0 and freqS [ i ] != 0 ) : NEW_LINE INDENT return " No " NEW_LINE DEDENT
elif ( freqS [ i ] == 0 and freqT [ i ] != 0 ) : NEW_LINE INDENT return " No " NEW_LINE DEDENT
elif ( freqT [ i ] != 0 and freqS [ i ] != ( freqT [ i ] * N ) ) : NEW_LINE INDENT return " No " NEW_LINE DEDENT
return " Yes " NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ " abc " , " abb " , " acc " ] NEW_LINE T = " abc " NEW_LINE N = len ( arr ) NEW_LINE print ( checkIfPossible ( N , arr , T ) ) NEW_LINE DEDENT
def groupsOfOnes ( S , N ) : NEW_LINE
count = 0 NEW_LINE
st = [ ] NEW_LINE
for i in range ( N ) : NEW_LINE
' NEW_LINE INDENT if ( S [ i ] == '1' ) : NEW_LINE INDENT st . append ( 1 ) NEW_LINE DEDENT DEDENT
else : NEW_LINE
if ( len ( st ) > 0 ) : NEW_LINE INDENT count += 1 NEW_LINE while ( len ( st ) > 0 ) : NEW_LINE INDENT del st [ - 1 ] NEW_LINE DEDENT DEDENT
if ( len ( st ) ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT
return count NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
S = "100110111" NEW_LINE N = len ( S ) NEW_LINE
print ( groupsOfOnes ( S , N ) ) NEW_LINE
def generatePalindrome ( S ) : NEW_LINE
Hash = { } NEW_LINE
for ch in S : NEW_LINE INDENT Hash [ ch ] = Hash . get ( ch , 0 ) + 1 NEW_LINE DEDENT
st = { } NEW_LINE
for i in range ( ord ( ' a ' ) , ord ( ' z ' ) + 1 ) : NEW_LINE
if ( chr ( i ) in Hash and Hash [ chr ( i ) ] == 2 ) : NEW_LINE
for j in range ( ord ( ' a ' ) , ord ( ' z ' ) + 1 ) : NEW_LINE
s = " " NEW_LINE if ( chr ( j ) in Hash and i != j ) : NEW_LINE INDENT s += chr ( i ) NEW_LINE s += chr ( j ) NEW_LINE s += chr ( i ) NEW_LINE DEDENT
st [ s ] = 1 NEW_LINE
if ( ( chr ( i ) in Hash ) and Hash [ chr ( i ) ] >= 3 ) : NEW_LINE
for j in range ( ord ( ' a ' ) , ord ( ' z ' ) + 1 ) : NEW_LINE
s = " " NEW_LINE
if ( chr ( j ) in Hash ) : NEW_LINE INDENT s += chr ( i ) NEW_LINE s += chr ( j ) NEW_LINE s += chr ( i ) NEW_LINE DEDENT
st [ s ] = 1 NEW_LINE
for ans in st : NEW_LINE INDENT print ( ans ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT S = " ddabdac " NEW_LINE generatePalindrome ( S ) NEW_LINE DEDENT
def countOccurrences ( S , X , Y ) : NEW_LINE
count = 0 NEW_LINE
N = len ( S ) NEW_LINE A = len ( X ) NEW_LINE B = len ( Y ) NEW_LINE
for i in range ( 0 , N ) : NEW_LINE
if ( S [ i : i + B ] == Y ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT
if ( S [ i : i + A ] == X ) : NEW_LINE INDENT print ( count , end = " ▁ " ) NEW_LINE DEDENT
S = " abcdefdefabc " NEW_LINE X = " abc " NEW_LINE Y = " def " NEW_LINE countOccurrences ( S , X , Y ) NEW_LINE
def DFA ( str , N ) : NEW_LINE
if ( N <= 1 ) : NEW_LINE INDENT print ( " No " ) NEW_LINE return NEW_LINE DEDENT
count = 0 NEW_LINE
if ( str [ 0 ] == ' C ' ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT
for i in range ( 1 , N ) : NEW_LINE
if ( str [ i ] == ' A ' or str [ i ] == ' B ' ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT else : NEW_LINE INDENT break NEW_LINE DEDENT else : NEW_LINE
print ( " No " ) NEW_LINE return NEW_LINE
if ( count == N ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT str = " CAABBAAB " NEW_LINE N = len ( str ) NEW_LINE DFA ( str , N ) NEW_LINE DEDENT
def minMaxDigits ( str , N ) : NEW_LINE
arr = [ 0 ] * N NEW_LINE for i in range ( N ) : NEW_LINE INDENT arr [ i ] = ( ord ( str [ i ] ) - ord ( '0' ) ) % 3 NEW_LINE DEDENT
zero = 0 NEW_LINE one = 0 NEW_LINE two = 0 NEW_LINE
for i in range ( N ) : NEW_LINE INDENT if ( arr [ i ] == 0 ) : NEW_LINE INDENT zero += 1 NEW_LINE DEDENT if ( arr [ i ] == 1 ) : NEW_LINE INDENT one += 1 NEW_LINE DEDENT if ( arr [ i ] == 2 ) : NEW_LINE INDENT two += 1 NEW_LINE DEDENT DEDENT
sum = 0 NEW_LINE for i in range ( N ) : NEW_LINE INDENT sum = ( sum + arr [ i ] ) % 3 NEW_LINE DEDENT
if ( sum == 0 ) : NEW_LINE INDENT print ( "0" , end = " ▁ " ) NEW_LINE DEDENT if ( sum == 1 ) : NEW_LINE INDENT if ( one and N > 1 ) : NEW_LINE INDENT print ( "1" , end = " ▁ " ) NEW_LINE DEDENT elif ( two > 1 and N > 2 ) : NEW_LINE INDENT print ( "2" , end = " ▁ " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " - 1" , end = " ▁ " ) NEW_LINE DEDENT DEDENT if ( sum == 2 ) : NEW_LINE INDENT if ( two and N > 1 ) : NEW_LINE INDENT print ( "1" , end = " ▁ " ) NEW_LINE DEDENT elif ( one > 1 and N > 2 ) : NEW_LINE INDENT print ( "2" , end = " ▁ " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " - 1" , end = " ▁ " ) NEW_LINE DEDENT DEDENT
if ( zero > 0 ) : NEW_LINE INDENT print ( N - 1 , end = " ▁ " ) NEW_LINE DEDENT elif ( one > 0 and two > 0 ) : NEW_LINE INDENT print ( N - 2 , end = " ▁ " ) NEW_LINE DEDENT elif ( one > 2 or two > 2 ) : NEW_LINE INDENT print ( N - 3 , end = " ▁ " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " - 1" , end = " ▁ " ) NEW_LINE DEDENT
str = "12345" NEW_LINE N = len ( str ) NEW_LINE
minMaxDigits ( str , N ) NEW_LINE
import sys NEW_LINE
def findMinimumChanges ( N , K , S ) : NEW_LINE
ans = 0 NEW_LINE
for i in range ( ( K + 1 ) // 2 ) : NEW_LINE
mp = { } NEW_LINE
for j in range ( i , N , K ) : NEW_LINE
mp [ S [ j ] ] = mp . get ( S [ j ] , 0 ) + 1 NEW_LINE
j = N - i - 1 NEW_LINE while ( j >= 0 ) : NEW_LINE
if ( ( K & 1 ) and ( i == K // 2 ) ) : NEW_LINE INDENT break NEW_LINE DEDENT
mp [ S [ j ] ] = mp . get ( S [ j ] , 0 ) + 1 NEW_LINE j -= K NEW_LINE
curr_max = - sys . maxsize - 1 NEW_LINE for key , value in mp . items ( ) : NEW_LINE INDENT curr_max = max ( curr_max , value ) NEW_LINE DEDENT
if ( ( K & 1 ) and ( i == K // 2 ) ) : NEW_LINE INDENT ans += ( N // K - curr_max ) NEW_LINE DEDENT
else : NEW_LINE INDENT ans += ( N // K * 2 - curr_max ) NEW_LINE DEDENT
return ans NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT S = " aabbcbbcb " NEW_LINE N = len ( S ) NEW_LINE K = 3 NEW_LINE DEDENT
print ( findMinimumChanges ( N , K , S ) ) NEW_LINE
def checkString ( s , K ) : NEW_LINE INDENT n = len ( s ) NEW_LINE DEDENT
dict = { } NEW_LINE for i in range ( n ) : NEW_LINE INDENT dict [ s [ i ] ] = i ; NEW_LINE DEDENT
st = set ( ) NEW_LINE for i in range ( n ) : NEW_LINE
st . add ( s [ i ] ) NEW_LINE
if len ( st ) > K : NEW_LINE INDENT print ( " Yes " ) NEW_LINE return NEW_LINE DEDENT
if dict [ s [ i ] ] == i : NEW_LINE INDENT st . remove ( s [ i ] ) NEW_LINE DEDENT print ( " No " ) NEW_LINE
s = " aabbcdca " NEW_LINE K = 2 NEW_LINE checkString ( s , K ) NEW_LINE
def distinct ( S , M ) : NEW_LINE INDENT count = 0 NEW_LINE DEDENT
for i in range ( len ( S ) ) : NEW_LINE
c = len ( set ( [ d for d in S [ i ] ] ) ) NEW_LINE
if ( c <= M ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT print ( count ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT S = [ " HERBIVORES " , " AEROPLANE " , " GEEKSFORGEEKS " ] NEW_LINE M = 7 NEW_LINE distinct ( S , M ) NEW_LINE DEDENT
def removeOddFrequencyCharacters ( s ) : NEW_LINE
m = dict ( ) NEW_LINE for i in s : NEW_LINE INDENT m [ i ] = m . get ( i , 0 ) + 1 NEW_LINE DEDENT
new_s = " " NEW_LINE
for i in s : NEW_LINE
if ( m [ i ] & 1 ) : NEW_LINE INDENT continue NEW_LINE DEDENT
new_s += i NEW_LINE
return new_s NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT str = " geeksforgeeks " NEW_LINE DEDENT
str = removeOddFrequencyCharacters ( str ) NEW_LINE print ( str ) NEW_LINE
def productAtKthLevel ( tree , k , i , level ) : NEW_LINE INDENT if ( tree [ i [ 0 ] ] == ' ( ' ) : NEW_LINE INDENT i [ 0 ] += 1 NEW_LINE DEDENT DEDENT
if ( tree [ i [ 0 ] ] == ' ) ' ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT product = 1 NEW_LINE
if ( level == k ) : NEW_LINE INDENT product = int ( tree [ i [ 0 ] ] ) NEW_LINE DEDENT
i [ 0 ] += 1 NEW_LINE leftproduct = productAtKthLevel ( tree , k , i , level + 1 ) NEW_LINE
i [ 0 ] += 1 NEW_LINE rightproduct = productAtKthLevel ( tree , k , i , level + 1 ) NEW_LINE
i [ 0 ] += 1 NEW_LINE return product * leftproduct * rightproduct NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT tree = " ( 0(5(6 ( ) ( ) ) ( 4 ( ) (9 ( ) ( ) ) ) ) ( 7(1 ( ) ( ) ) ( 3 ( ) ( ) ) ) ) " NEW_LINE k = 2 NEW_LINE i = [ 0 ] NEW_LINE print ( productAtKthLevel ( tree , k , i , 0 ) ) NEW_LINE DEDENT
def findMostOccurringChar ( string ) : NEW_LINE
hash = [ 0 ] * 26 ; NEW_LINE
for i in range ( len ( string ) ) : NEW_LINE
for j in range ( len ( string [ i ] ) ) : NEW_LINE
hash [ ord ( string [ i ] [ j ] ) - ord ( ' a ' ) ] += 1 ; NEW_LINE
max = 0 ; NEW_LINE for i in range ( 26 ) : NEW_LINE INDENT max = i if hash [ i ] > hash [ max ] else max ; NEW_LINE DEDENT print ( ( chr ) ( max + 97 ) ) ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
string = [ ] ; NEW_LINE string . append ( " animal " ) ; NEW_LINE string . append ( " zebra " ) ; NEW_LINE string . append ( " lion " ) ; NEW_LINE string . append ( " giraffe " ) ; NEW_LINE findMostOccurringChar ( string ) ; NEW_LINE
def isPalindrome ( num ) : NEW_LINE
s = str ( num ) NEW_LINE
low = 0 NEW_LINE high = len ( s ) - 1 NEW_LINE while ( low < high ) : NEW_LINE
if ( s [ low ] != s [ high ] ) : NEW_LINE INDENT return False NEW_LINE DEDENT
low += 1 NEW_LINE high -= 1 NEW_LINE return True NEW_LINE
n = 123.321 NEW_LINE if ( isPalindrome ( n ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
import sys NEW_LINE MAX = 26 ; NEW_LINE
def maxSubStr ( str1 , len1 , str2 , len2 ) : NEW_LINE
if ( len1 > len2 ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT
freq1 = [ 0 ] * MAX ; NEW_LINE for i in range ( len1 ) : NEW_LINE INDENT freq1 [ ord ( str1 [ i ] ) - ord ( ' a ' ) ] += 1 ; NEW_LINE DEDENT
freq2 = [ 0 ] * MAX ; NEW_LINE for i in range ( len2 ) : NEW_LINE INDENT freq2 [ ord ( str2 [ i ] ) - ord ( ' a ' ) ] += 1 ; NEW_LINE DEDENT
minPoss = sys . maxsize ; NEW_LINE for i in range ( MAX ) : NEW_LINE
if ( freq1 [ i ] == 0 ) : NEW_LINE INDENT continue ; NEW_LINE DEDENT
if ( freq1 [ i ] > freq2 [ i ] ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT
minPoss = min ( minPoss , freq2 [ i ] / freq1 [ i ] ) ; NEW_LINE return int ( minPoss ) ; NEW_LINE
str1 = " geeks " ; str2 = " gskefrgoekees " ; NEW_LINE len1 = len ( str1 ) ; NEW_LINE len2 = len ( str2 ) ; NEW_LINE print ( maxSubStr ( str1 , len1 , str2 , len2 ) ) ; NEW_LINE
def cntWays ( string , n ) : NEW_LINE INDENT x = n + 1 ; NEW_LINE ways = x * x * ( x * x - 1 ) // 12 ; NEW_LINE return ways ; NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT string = " ab " ; NEW_LINE n = len ( string ) ; NEW_LINE print ( cntWays ( string , n ) ) ; NEW_LINE DEDENT
import sys NEW_LINE
uSet = set ( ) ; NEW_LINE
minCnt = sys . maxsize ; NEW_LINE
def findSubStr ( string , cnt , start ) : NEW_LINE INDENT global minCnt ; NEW_LINE DEDENT
if ( start == len ( string ) ) : NEW_LINE
minCnt = min ( cnt , minCnt ) ; NEW_LINE
for length in range ( 1 , len ( string ) - start + 1 ) : NEW_LINE
subStr = string [ start : start + length ] ; NEW_LINE
if subStr in uSet : NEW_LINE
findSubStr ( string , cnt + 1 , start + length ) ; NEW_LINE
def findMinSubStr ( arr , n , string ) : NEW_LINE
for i in range ( n ) : NEW_LINE INDENT uSet . add ( arr [ i ] ) ; NEW_LINE DEDENT
findSubStr ( string , 0 , 0 ) ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT string = "123456" ; NEW_LINE arr = [ "1" , "12345" , "2345" , "56" , "23" , "456" ] ; NEW_LINE n = len ( arr ) ; NEW_LINE findMinSubStr ( arr , n , string ) ; NEW_LINE print ( minCnt ) ; NEW_LINE DEDENT
def countSubStr ( s , n ) : NEW_LINE INDENT c1 = 0 ; c2 = 0 ; NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE
if ( s [ i : i + 5 ] == " geeks " ) : NEW_LINE INDENT c1 += 1 ; NEW_LINE DEDENT
if ( s [ i : i + 3 ] == " for " ) : NEW_LINE INDENT c2 = c2 + c1 ; NEW_LINE DEDENT return c2 ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT s = " geeksforgeeksisforgeeks " ; NEW_LINE n = len ( s ) ; NEW_LINE print ( countSubStr ( s , n ) ) ; NEW_LINE DEDENT
string = " { [ ( ) ] } [ ] " NEW_LINE
lst1 = [ ' { ' , ' ( ' , ' [ ' ] NEW_LINE
lst2 = [ ' } ' , ' ) ' , ' ] ' ] NEW_LINE
lst = [ ] NEW_LINE
Dict = { ' ) ' : ' ( ' , ' } ' : ' { ' , ' ] ' : ' [ ' } NEW_LINE a = b = c = 0 NEW_LINE
if string [ 0 ] in lst2 : NEW_LINE INDENT print ( 1 ) NEW_LINE DEDENT else : NEW_LINE
for i in range ( 0 , len ( string ) ) : NEW_LINE INDENT if string [ i ] in lst1 : NEW_LINE INDENT lst . append ( string [ i ] ) NEW_LINE k = i + 2 NEW_LINE DEDENT else : NEW_LINE DEDENT
if len ( lst ) == 0 and ( string [ i ] in lst2 ) : NEW_LINE INDENT print ( i + 1 ) NEW_LINE c = 1 NEW_LINE break NEW_LINE DEDENT else : NEW_LINE
if Dict [ string [ i ] ] == lst [ len ( lst ) - 1 ] : NEW_LINE INDENT lst . pop ( ) NEW_LINE DEDENT else : NEW_LINE
print ( i + 1 ) NEW_LINE a = 1 NEW_LINE break NEW_LINE
if len ( lst ) == 0 and c == 0 : NEW_LINE INDENT print ( 0 ) NEW_LINE b = 1 NEW_LINE DEDENT if a == 0 and b == 0 and c == 0 : NEW_LINE INDENT print ( k ) NEW_LINE DEDENT
MAX = 26 NEW_LINE
def encryptstrr ( strr , n , x ) : NEW_LINE
x = x % MAX NEW_LINE arr = list ( strr ) NEW_LINE
freq = [ 0 ] * MAX NEW_LINE for i in range ( n ) : NEW_LINE INDENT freq [ ord ( arr [ i ] ) - ord ( ' a ' ) ] += 1 NEW_LINE DEDENT for i in range ( n ) : NEW_LINE
if ( freq [ ord ( arr [ i ] ) - ord ( ' a ' ) ] % 2 == 0 ) : NEW_LINE INDENT pos = ( ord ( arr [ i ] ) - ord ( ' a ' ) + x ) % MAX NEW_LINE arr [ i ] = chr ( pos + ord ( ' a ' ) ) NEW_LINE DEDENT
else : NEW_LINE INDENT pos = ( ord ( arr [ i ] ) - ord ( ' a ' ) - x ) NEW_LINE if ( pos < 0 ) : NEW_LINE INDENT pos += MAX NEW_LINE DEDENT arr [ i ] = chr ( pos + ord ( ' a ' ) ) NEW_LINE DEDENT
return " " . join ( arr ) NEW_LINE
s = " abcda " NEW_LINE n = len ( s ) NEW_LINE x = 3 NEW_LINE print ( encryptstrr ( s , n , x ) ) NEW_LINE
def isPossible ( Str ) : NEW_LINE
freq = dict ( ) NEW_LINE
max_freq = 0 NEW_LINE for j in range ( len ( Str ) ) : NEW_LINE INDENT freq [ Str [ j ] ] = freq . get ( Str [ j ] , 0 ) + 1 NEW_LINE if ( freq [ Str [ j ] ] > max_freq ) : NEW_LINE INDENT max_freq = freq [ Str [ j ] ] NEW_LINE DEDENT DEDENT
if ( max_freq <= ( len ( Str ) - max_freq + 1 ) ) : NEW_LINE INDENT return True NEW_LINE DEDENT return False NEW_LINE
Str = " geeksforgeeks " NEW_LINE if ( isPossible ( Str ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
def printUncommon ( str1 , str2 ) : NEW_LINE INDENT a1 = 0 ; a2 = 0 ; NEW_LINE for i in range ( len ( str1 ) ) : NEW_LINE DEDENT
ch = ord ( str1 [ i ] ) - ord ( ' a ' ) ; NEW_LINE
a1 = a1 | ( 1 << ch ) ; NEW_LINE for i in range ( len ( str2 ) ) : NEW_LINE
ch = ord ( str2 [ i ] ) - ord ( ' a ' ) ; NEW_LINE
a2 = a2 | ( 1 << ch ) ; NEW_LINE
ans = a1 ^ a2 ; NEW_LINE i = 0 ; NEW_LINE while ( i < 26 ) : NEW_LINE INDENT if ( ans % 2 == 1 ) : NEW_LINE INDENT print ( chr ( ord ( ' a ' ) + i ) , end = " " ) ; NEW_LINE DEDENT ans = ans // 2 ; NEW_LINE i += 1 ; NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT str1 = " geeksforgeeks " ; NEW_LINE str2 = " geeksquiz " ; NEW_LINE printUncommon ( str1 , str2 ) ; NEW_LINE DEDENT
def countMinReversals ( expr ) : NEW_LINE INDENT length = len ( expr ) NEW_LINE DEDENT
if length % 2 : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
ans = 0 NEW_LINE
open = 0 NEW_LINE
close = 0 NEW_LINE for i in range ( 0 , length ) : NEW_LINE
if expr [ i ] == " " : NEW_LINE INDENT open += 1 NEW_LINE DEDENT
else : NEW_LINE INDENT if not open : NEW_LINE INDENT close += 1 NEW_LINE DEDENT else : NEW_LINE INDENT open -= 1 NEW_LINE DEDENT DEDENT ans = ( close // 2 ) + ( open // 2 ) NEW_LINE
close %= 2 NEW_LINE open %= 2 NEW_LINE if close > 0 : NEW_LINE INDENT ans += 2 NEW_LINE DEDENT return ans NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT expr = " } } { { " NEW_LINE print ( countMinReversals ( expr ) ) NEW_LINE DEDENT
def totalPairs ( s1 , s2 ) : NEW_LINE INDENT a1 = 0 ; b1 = 0 ; NEW_LINE DEDENT
for i in range ( len ( s1 ) ) : NEW_LINE INDENT if ( ord ( s1 [ i ] ) % 2 != 0 ) : NEW_LINE INDENT a1 += 1 ; NEW_LINE DEDENT else : NEW_LINE INDENT b1 += 1 ; NEW_LINE DEDENT DEDENT a2 = 0 ; b2 = 0 ; NEW_LINE
for i in range ( len ( s2 ) ) : NEW_LINE INDENT if ( ord ( s2 [ i ] ) % 2 != 0 ) : NEW_LINE INDENT a2 += 1 ; NEW_LINE DEDENT else : NEW_LINE INDENT b2 += 1 ; NEW_LINE DEDENT DEDENT
return ( ( a1 * a2 ) + ( b1 * b2 ) ) ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT s1 = " geeks " ; NEW_LINE s2 = " for " ; NEW_LINE print ( totalPairs ( s1 , s2 ) ) ; NEW_LINE DEDENT
def prefixOccurrences ( str1 ) : NEW_LINE INDENT c = str1 [ 0 ] NEW_LINE countc = 0 NEW_LINE DEDENT
for i in range ( len ( str1 ) ) : NEW_LINE INDENT if ( str1 [ i ] == c ) : NEW_LINE INDENT countc += 1 NEW_LINE DEDENT DEDENT return countc NEW_LINE
str1 = " abbcdabbcd " NEW_LINE print ( prefixOccurrences ( str1 ) ) NEW_LINE
def minOperations ( s , t , n ) : NEW_LINE INDENT ct0 = 0 NEW_LINE ct1 = 0 NEW_LINE for i in range ( n ) : NEW_LINE DEDENT
if ( s [ i ] == t [ i ] ) : NEW_LINE INDENT continue NEW_LINE DEDENT
if ( s [ i ] == '0' ) : NEW_LINE INDENT ct0 += 1 NEW_LINE DEDENT
else : NEW_LINE INDENT ct1 += 1 NEW_LINE DEDENT return max ( ct0 , ct1 ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT s = "010" NEW_LINE t = "101" NEW_LINE n = len ( s ) NEW_LINE print ( minOperations ( s , t , n ) ) NEW_LINE DEDENT
def decryptString ( str , n ) : NEW_LINE
i = 0 NEW_LINE jump = 1 NEW_LINE decryptedStr = " " NEW_LINE while ( i < n ) : NEW_LINE INDENT decryptedStr += str [ i ] ; NEW_LINE i += jump NEW_LINE DEDENT
jump += 1 NEW_LINE return decryptedStr NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT str = " geeeeekkkksssss " NEW_LINE n = len ( str ) NEW_LINE print ( decryptString ( str , n ) ) NEW_LINE DEDENT
def bitToBeFlipped ( s ) : NEW_LINE
last = s [ len ( s ) - 1 ] NEW_LINE first = s [ 0 ] NEW_LINE
if ( last == first ) : NEW_LINE INDENT if ( last == '0' ) : NEW_LINE INDENT return '1' NEW_LINE DEDENT else : NEW_LINE INDENT return '0' NEW_LINE DEDENT DEDENT
elif ( last != first ) : NEW_LINE INDENT return last NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT s = "1101011000" NEW_LINE print ( bitToBeFlipped ( s ) ) NEW_LINE DEDENT
def SieveofEratosthenes ( prime , p_size ) : NEW_LINE
prime [ 0 ] = False NEW_LINE prime [ 1 ] = False NEW_LINE for p in range ( 2 , p_size + 1 ) : NEW_LINE
if prime [ p ] : NEW_LINE
for i in range ( p * 2 , p_size + 1 , p ) : NEW_LINE INDENT prime [ i ] = False NEW_LINE DEDENT
def sumProdOfPrimeFreq ( s ) : NEW_LINE INDENT prime = [ True ] * ( len ( s ) + 2 ) NEW_LINE SieveofEratosthenes ( prime , len ( s ) + 1 ) NEW_LINE i = 0 NEW_LINE j = 0 NEW_LINE DEDENT
m = dict ( ) NEW_LINE for i in range ( len ( s ) ) : NEW_LINE INDENT m [ s [ i ] ] = ( m [ s [ i ] ] + 1 ) if s [ i ] in m else 1 NEW_LINE DEDENT s = 0 NEW_LINE product = 1 NEW_LINE
for it in m : NEW_LINE
if prime [ m [ it ] ] : NEW_LINE INDENT s += m [ it ] NEW_LINE product *= m [ it ] NEW_LINE DEDENT print ( " Sum ▁ = " , s ) NEW_LINE print ( " Product ▁ = " , product ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT s = " geeksforgeeks " NEW_LINE sumProdOfPrimeFreq ( s ) NEW_LINE DEDENT
from collections import defaultdict NEW_LINE
def multipleOrFactor ( s1 , s2 ) : NEW_LINE
m1 = defaultdict ( lambda : 0 ) NEW_LINE m2 = defaultdict ( lambda : 0 ) NEW_LINE for i in range ( 0 , len ( s1 ) ) : NEW_LINE INDENT m1 [ s1 [ i ] ] += 1 NEW_LINE DEDENT for i in range ( 0 , len ( s2 ) ) : NEW_LINE INDENT m2 [ s2 [ i ] ] += 1 NEW_LINE DEDENT for it in m1 : NEW_LINE
if it not in m2 : NEW_LINE INDENT continue NEW_LINE DEDENT
if ( m2 [ it ] % m1 [ it ] == 0 or m1 [ it ] % m2 [ it ] == 0 ) : NEW_LINE INDENT continue NEW_LINE DEDENT
else : NEW_LINE INDENT return False NEW_LINE DEDENT return True NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT s1 = " geeksforgeeks " NEW_LINE s2 = " geeks " NEW_LINE if multipleOrFactor ( s1 , s2 ) : print ( " YES " ) NEW_LINE else : print ( " NO " ) NEW_LINE DEDENT
def solve ( s ) : NEW_LINE
m = dict ( ) NEW_LINE for i in range ( len ( s ) ) : NEW_LINE INDENT if s [ i ] in m : NEW_LINE INDENT m [ s [ i ] ] = m [ s [ i ] ] + 1 NEW_LINE DEDENT else : NEW_LINE INDENT m [ s [ i ] ] = 1 NEW_LINE DEDENT DEDENT
new_string = " " NEW_LINE
for i in range ( len ( s ) ) : NEW_LINE
if m [ s [ i ] ] % 2 == 0 : NEW_LINE INDENT continue NEW_LINE DEDENT
new_string = new_string + s [ i ] NEW_LINE
print ( new_string ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT s = " aabbbddeeecc " NEW_LINE DEDENT
solve ( s ) NEW_LINE
def isPalindrome ( string ) : NEW_LINE INDENT i = 0 ; j = len ( string ) - 1 ; NEW_LINE DEDENT
while ( i < j ) : NEW_LINE
if ( string [ i ] != string [ j ] ) : NEW_LINE INDENT return False ; NEW_LINE DEDENT i += 1 ; NEW_LINE j -= 1 ; NEW_LINE
return True ; NEW_LINE
def removePalinWords ( string ) : NEW_LINE
' NEW_LINE INDENT final_str = " " ; word = " " ; NEW_LINE DEDENT
' NEW_LINE INDENT string = string + " ▁ " ; NEW_LINE n = len ( string ) ; NEW_LINE DEDENT
' NEW_LINE INDENT for i in range ( n ) : NEW_LINE DEDENT
if ( string [ i ] != ' ▁ ' ) : NEW_LINE INDENT word = word + string [ i ] ; NEW_LINE DEDENT else : NEW_LINE
' NEW_LINE INDENT if ( not ( isPalindrome ( word ) ) ) : NEW_LINE INDENT final_str += word + " ▁ " ; NEW_LINE DEDENT DEDENT
word = " " ; NEW_LINE
return final_str ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT string = " Text ▁ contains ▁ malayalam ▁ and ▁ level ▁ words " ; NEW_LINE print ( removePalinWords ( string ) ) ; NEW_LINE DEDENT
def findSubSequence ( s , num ) : NEW_LINE
res = 0 NEW_LINE
i = 0 NEW_LINE while ( num ) : NEW_LINE
if ( num & 1 ) : NEW_LINE INDENT res += ord ( s [ i ] ) - ord ( '0' ) NEW_LINE DEDENT i += 1 NEW_LINE
num = num >> 1 NEW_LINE return res NEW_LINE
def combinedSum ( s ) : NEW_LINE
n = len ( s ) NEW_LINE
c_sum = 0 NEW_LINE
ran = ( 1 << n ) - 1 NEW_LINE
for i in range ( ran + 1 ) : NEW_LINE INDENT c_sum += findSubSequence ( s , i ) NEW_LINE DEDENT
return c_sum NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT s = "123" NEW_LINE print ( combinedSum ( s ) ) NEW_LINE DEDENT
MAX_CHAR = 26 NEW_LINE
def findSubsequence ( stri , k ) : NEW_LINE
a = [ 0 ] * MAX_CHAR ; NEW_LINE
for i in range ( len ( stri ) ) : NEW_LINE INDENT a [ ord ( stri [ i ] ) - ord ( ' a ' ) ] += 1 NEW_LINE DEDENT
for i in range ( len ( stri ) ) : NEW_LINE INDENT if a [ ord ( stri [ i ] ) - ord ( ' a ' ) ] >= k : NEW_LINE INDENT print ( stri [ i ] , end = ' ' ) NEW_LINE DEDENT DEDENT
k = 2 NEW_LINE findSubsequence ( " geeksforgeeks " , k ) NEW_LINE
def convert ( str ) : NEW_LINE
w = " " NEW_LINE z = " " ; NEW_LINE
str = str . upper ( ) + " ▁ " ; NEW_LINE for i in range ( len ( str ) ) : NEW_LINE
ch = str [ i ] ; NEW_LINE if ( ch != ' ▁ ' ) : NEW_LINE INDENT w = w + ch ; NEW_LINE DEDENT else : NEW_LINE
z = ( z + ( w [ 0 ] ) . lower ( ) + w [ 1 : len ( w ) ] + " ▁ " ) ; NEW_LINE w = " " ; NEW_LINE return z ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT str = " I ▁ got ▁ intern ▁ at ▁ geeksforgeeks " ; NEW_LINE print ( convert ( str ) ) ; NEW_LINE DEDENT
def isVowel ( c ) : NEW_LINE INDENT return ( c == ' a ' or c == ' e ' or c == ' i ' or c == ' o ' or c == ' u ' ) NEW_LINE DEDENT
def encryptString ( s , n , k ) : NEW_LINE
cv = [ 0 for i in range ( n ) ] NEW_LINE cc = [ 0 for i in range ( n ) ] NEW_LINE if ( isVowel ( s [ 0 ] ) ) : NEW_LINE INDENT cv [ 0 ] = 1 NEW_LINE DEDENT else : NEW_LINE INDENT cc [ 0 ] = 1 NEW_LINE DEDENT
for i in range ( 1 , n ) : NEW_LINE INDENT cv [ i ] = cv [ i - 1 ] + isVowel ( s [ i ] ) NEW_LINE cc [ i ] = cc [ i - 1 ] + ( isVowel ( s [ i ] ) == False ) NEW_LINE DEDENT ans = " " NEW_LINE prod = 0 NEW_LINE prod = cc [ k - 1 ] * cv [ k - 1 ] NEW_LINE ans += str ( prod ) NEW_LINE
for i in range ( k , len ( s ) ) : NEW_LINE INDENT prod = ( ( cc [ i ] - cc [ i - k ] ) * ( cv [ i ] - cv [ i - k ] ) ) NEW_LINE ans += str ( prod ) NEW_LINE DEDENT return ans NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT s = " hello " NEW_LINE n = len ( s ) NEW_LINE k = 2 NEW_LINE print ( encryptString ( s , n , k ) ) NEW_LINE DEDENT
def countOccurrences ( str , word ) : NEW_LINE
a = str . split ( " ▁ " ) NEW_LINE
count = 0 NEW_LINE for i in range ( 0 , len ( a ) ) : NEW_LINE
if ( word == a [ i ] ) : NEW_LINE count = count + 1 NEW_LINE return count NEW_LINE
str = " GeeksforGeeks ▁ A ▁ computer ▁ science ▁ portal ▁ for ▁ geeks ▁ " NEW_LINE word = " portal " NEW_LINE print ( countOccurrences ( str , word ) ) NEW_LINE
def printInitials ( name ) : NEW_LINE INDENT if ( len ( name ) == 0 ) : NEW_LINE INDENT return NEW_LINE DEDENT DEDENT
words = name . split ( " ▁ " ) NEW_LINE for word in words : NEW_LINE INDENT print ( word [ 0 ] . upper ( ) , end = " ▁ " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT name = " prabhat ▁ kumar ▁ singh " NEW_LINE printInitials ( name ) NEW_LINE DEDENT
def permute ( inp ) : NEW_LINE INDENT n = len ( inp ) NEW_LINE DEDENT
mx = 1 << n NEW_LINE
inp = inp . lower ( ) NEW_LINE
for i in range ( mx ) : NEW_LINE
combination = [ k for k in inp ] NEW_LINE for j in range ( n ) : NEW_LINE INDENT if ( ( ( i >> j ) & 1 ) == 1 ) : NEW_LINE INDENT combination [ j ] = inp [ j ] . upper ( ) NEW_LINE DEDENT DEDENT temp = " " NEW_LINE
for i in combination : NEW_LINE INDENT temp += i NEW_LINE DEDENT print temp , NEW_LINE
permute ( " ABC " ) NEW_LINE
def printString ( str , ch , count ) : NEW_LINE INDENT occ , i = 0 , 0 NEW_LINE DEDENT
if ( count == 0 ) : NEW_LINE INDENT print ( str ) NEW_LINE DEDENT
for i in range ( len ( str ) ) : NEW_LINE
if ( str [ i ] == ch ) : NEW_LINE INDENT occ += 1 NEW_LINE DEDENT
if ( occ == count ) : NEW_LINE INDENT break NEW_LINE DEDENT
if ( i < len ( str ) - 1 ) : NEW_LINE INDENT print ( str [ i + 1 : len ( str ) - i + 2 ] ) NEW_LINE DEDENT
else : NEW_LINE INDENT print ( " Empty ▁ string " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT str = " geeks ▁ for ▁ geeks " NEW_LINE printString ( str , ' e ' , 2 ) NEW_LINE DEDENT
def isVowel ( c ) : NEW_LINE INDENT return ( c == ' a ' or c == ' A ' or c == ' e ' or c == ' E ' or c == ' i ' or c == ' I ' or c == ' o ' or c == ' O ' or c == ' u ' or c == ' U ' ) NEW_LINE DEDENT
def reverseVowel ( str ) : NEW_LINE
i = 0 NEW_LINE j = len ( str ) - 1 NEW_LINE while ( i < j ) : NEW_LINE INDENT if not isVowel ( str [ i ] ) : NEW_LINE INDENT i += 1 NEW_LINE continue NEW_LINE DEDENT if ( not isVowel ( str [ j ] ) ) : NEW_LINE INDENT j -= 1 NEW_LINE continue NEW_LINE DEDENT DEDENT
str [ i ] , str [ j ] = str [ j ] , str [ i ] NEW_LINE i += 1 NEW_LINE j -= 1 NEW_LINE return str NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT str = " hello ▁ world " NEW_LINE print ( * reverseVowel ( list ( str ) ) , sep = " " ) NEW_LINE DEDENT
def isPalindrome ( str ) : NEW_LINE
l = 0 NEW_LINE h = len ( str ) - 1 NEW_LINE
while ( h > l ) : NEW_LINE INDENT if ( str [ l ] != str [ h ] ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT l = l + 1 NEW_LINE h = h - 1 NEW_LINE DEDENT return 1 NEW_LINE
def minRemovals ( str ) : NEW_LINE
if ( str [ 0 ] == ' ' ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
if ( isPalindrome ( str ) ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT
return 2 NEW_LINE
print ( minRemovals ( "010010" ) ) NEW_LINE print ( minRemovals ( "0100101" ) ) NEW_LINE
def power ( x , y , p ) : NEW_LINE
res = 1 ; NEW_LINE
x = x % p ; NEW_LINE while ( y > 0 ) : NEW_LINE
if ( y and 1 ) : NEW_LINE INDENT res = ( res * x ) % p ; NEW_LINE DEDENT
y = y >> 1 ; NEW_LINE x = ( x * x ) % p ; NEW_LINE return res ; NEW_LINE
def findModuloByM ( X , N , M ) : NEW_LINE
if ( N < 6 ) : NEW_LINE
temp = chr ( 48 + X ) * N NEW_LINE
res = int ( temp ) % M ; NEW_LINE return res ; NEW_LINE
if ( N % 2 == 0 ) : NEW_LINE
half = findModuloByM ( X , N // 2 , M ) % M ; NEW_LINE
res = ( half * power ( 10 , N // 2 , M ) + half ) % M ; NEW_LINE return res ; NEW_LINE else : NEW_LINE
half = findModuloByM ( X , N // 2 , M ) % M ; NEW_LINE
res = ( half * power ( 10 , N // 2 + 1 , M ) + half * 10 + X ) % M ; NEW_LINE return res ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT X = 6 ; N = 14 ; M = 9 ; NEW_LINE DEDENT
print ( findModuloByM ( X , N , M ) ) ; NEW_LINE
from math import sqrt NEW_LINE
class circle : NEW_LINE INDENT def __init__ ( self , a , b , c ) : NEW_LINE INDENT self . x = a NEW_LINE self . y = b NEW_LINE self . r = c NEW_LINE DEDENT DEDENT
def check ( C ) : NEW_LINE
C1C2 = sqrt ( ( C [ 1 ] . x - C [ 0 ] . x ) * ( C [ 1 ] . x - C [ 0 ] . x ) + ( C [ 1 ] . y - C [ 0 ] . y ) * ( C [ 1 ] . y - C [ 0 ] . y ) ) NEW_LINE
flag = 0 NEW_LINE
if ( C1C2 < ( C [ 0 ] . r + C [ 1 ] . r ) ) : NEW_LINE
if ( ( C [ 0 ] . x + C [ 1 ] . x ) == 2 * C [ 2 ] . x and ( C [ 0 ] . y + C [ 1 ] . y ) == 2 * C [ 2 ] . y ) : NEW_LINE
flag = 1 NEW_LINE
return flag NEW_LINE
def IsFairTriplet ( c ) : NEW_LINE INDENT f = False NEW_LINE DEDENT
f |= check ( c ) NEW_LINE for i in range ( 2 ) : NEW_LINE INDENT c [ 0 ] , c [ 2 ] = c [ 2 ] , c [ 0 ] NEW_LINE DEDENT
f |= check ( c ) NEW_LINE return f NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT C = [ circle ( 0 , 0 , 0 ) for i in range ( 3 ) ] NEW_LINE C [ 0 ] = circle ( 0 , 0 , 8 ) NEW_LINE C [ 1 ] = circle ( 0 , 10 , 6 ) NEW_LINE C [ 2 ] = circle ( 0 , 5 , 5 ) NEW_LINE if ( IsFairTriplet ( C ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT DEDENT
import math NEW_LINE
def eccHyperbola ( A , B ) : NEW_LINE
r = B * B / A * A NEW_LINE
r += 1 NEW_LINE
return math . sqrt ( r ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT A = 3.0 NEW_LINE B = 2.0 NEW_LINE print ( eccHyperbola ( A , B ) ) NEW_LINE DEDENT
from math import sqrt NEW_LINE
def calculateArea ( A , B , C , D ) : NEW_LINE
S = ( A + B + C + D ) // 2 NEW_LINE
area = sqrt ( ( S - A ) * ( S - B ) * ( S - C ) * ( S - D ) ) NEW_LINE
return area NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT A = 10 NEW_LINE B = 15 NEW_LINE C = 20 NEW_LINE D = 25 NEW_LINE print ( round ( calculateArea ( A , B , C , D ) , 3 ) ) NEW_LINE DEDENT
def triangleArea ( a , b ) : NEW_LINE
ratio = b / a NEW_LINE
print ( ratio ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = 1 NEW_LINE b = 2 NEW_LINE triangleArea ( a , b ) NEW_LINE DEDENT
from math import sqrt NEW_LINE
def distance ( m , n , p , q ) : NEW_LINE INDENT return ( sqrt ( pow ( n - m , 2 ) + pow ( q - p , 2 ) * 1.0 ) ) NEW_LINE DEDENT
def Excenters ( x1 , y1 , x2 , y2 , x3 , y3 ) : NEW_LINE
a = distance ( x2 , x3 , y2 , y3 ) NEW_LINE b = distance ( x3 , x1 , y3 , y1 ) NEW_LINE c = distance ( x1 , x2 , y1 , y2 ) NEW_LINE
excenter = [ [ 0 , 0 ] for i in range ( 4 ) ] NEW_LINE
excenter [ 1 ] [ 0 ] = ( ( - ( a * x1 ) + ( b * x2 ) + ( c * x3 ) ) // ( - a + b + c ) ) NEW_LINE excenter [ 1 ] [ 1 ] = ( ( - ( a * y1 ) + ( b * y2 ) + ( c * y3 ) ) // ( - a + b + c ) ) NEW_LINE
excenter [ 2 ] [ 0 ] = ( ( ( a * x1 ) - ( b * x2 ) + ( c * x3 ) ) // ( a - b + c ) ) NEW_LINE excenter [ 2 ] [ 1 ] = ( ( ( a * y1 ) - ( b * y2 ) + ( c * y3 ) ) // ( a - b + c ) ) NEW_LINE
excenter [ 3 ] [ 0 ] = ( ( ( a * x1 ) + ( b * x2 ) - ( c * x3 ) ) // ( a + b - c ) ) NEW_LINE excenter [ 3 ] [ 1 ] = ( ( ( a * y1 ) + ( b * y2 ) - ( c * y3 ) ) // ( a + b - c ) ) NEW_LINE
for i in range ( 1 , 4 ) : NEW_LINE INDENT print ( int ( excenter [ i ] [ 0 ] ) , int ( excenter [ i ] [ 1 ] ) ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT x1 = 0 NEW_LINE x2 = 3 NEW_LINE x3 = 0 NEW_LINE y1 = 0 NEW_LINE y2 = 0 NEW_LINE y3 = 4 NEW_LINE Excenters ( x1 , y1 , x2 , y2 , x3 , y3 ) NEW_LINE DEDENT
import math NEW_LINE
def findHeight ( p1 , p2 , b , c ) : NEW_LINE INDENT a = max ( p1 , p2 ) - min ( p1 , p2 ) NEW_LINE DEDENT
s = ( a + b + c ) // 2 NEW_LINE
area = math . sqrt ( s * ( s - a ) * ( s - b ) * ( s - c ) ) NEW_LINE
height = ( area * 2 ) / a NEW_LINE
print ( " Height ▁ is : ▁ " , height ) NEW_LINE
p1 = 25 NEW_LINE p2 = 10 NEW_LINE a = 14 NEW_LINE b = 13 NEW_LINE findHeight ( p1 , p2 , a , b ) NEW_LINE
def Icositetragonal_num ( n ) : NEW_LINE
return ( 22 * n * n - 20 * n ) / 2 NEW_LINE
n = 3 NEW_LINE print ( int ( Icositetragonal_num ( n ) ) ) NEW_LINE n = 10 NEW_LINE print ( int ( Icositetragonal_num ( n ) ) ) NEW_LINE
def area_of_circle ( m , n ) : NEW_LINE
square_of_radius = ( m * n ) / 4 NEW_LINE area = ( 3.141 * square_of_radius ) NEW_LINE return area NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 10 NEW_LINE m = 30 NEW_LINE print ( area_of_circle ( m , n ) ) NEW_LINE DEDENT
def area ( R ) : NEW_LINE
base = 1.732 * R NEW_LINE height = ( 3 / 2 ) * R NEW_LINE
area = ( ( 1 / 2 ) * base * height ) NEW_LINE return area NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT R = 7 NEW_LINE print ( area ( R ) ) NEW_LINE DEDENT
def circlearea ( R ) : NEW_LINE
if ( R < 0 ) : NEW_LINE INDENT return - 1 ; NEW_LINE DEDENT
a = ( 3.14 * R * R ) / 4 ; NEW_LINE return a ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT R = 2 ; NEW_LINE print ( circlearea ( R ) ) ; NEW_LINE DEDENT
def countPairs ( P , Q , N , M ) : NEW_LINE
A = [ 0 ] * 2 NEW_LINE B = [ 0 ] * 2 NEW_LINE
for i in range ( N ) : NEW_LINE INDENT A [ P [ i ] % 2 ] += 1 NEW_LINE DEDENT
for i in range ( M ) : NEW_LINE INDENT B [ Q [ i ] % 2 ] += 1 NEW_LINE DEDENT
return ( A [ 0 ] * B [ 0 ] + A [ 1 ] * B [ 1 ] ) NEW_LINE
P = [ 1 , 3 , 2 ] NEW_LINE Q = [ 3 , 0 ] NEW_LINE N = len ( P ) NEW_LINE M = len ( Q ) NEW_LINE print ( countPairs ( P , Q , N , M ) ) NEW_LINE
def countIntersections ( n ) : NEW_LINE INDENT return n * ( n - 1 ) // 2 NEW_LINE DEDENT
n = 3 NEW_LINE print ( countIntersections ( n ) ) NEW_LINE
import math NEW_LINE PI = 3.14159 NEW_LINE
def areaOfTriangle ( d ) : NEW_LINE
c = 1.618 * d NEW_LINE s = ( d + c + c ) / 2 NEW_LINE
area = math . sqrt ( s * ( s - c ) * ( s - c ) * ( s - d ) ) NEW_LINE
return 5 * area NEW_LINE
def areaOfRegPentagon ( d ) : NEW_LINE INDENT global PI NEW_LINE DEDENT
cal = 4 * math . tan ( PI / 5 ) NEW_LINE area = ( 5 * d * d ) / cal NEW_LINE
return area NEW_LINE
def areaOfPentagram ( d ) : NEW_LINE
return areaOfRegPentagon ( d ) + areaOfTriangle ( d ) NEW_LINE
d = 5 NEW_LINE print ( areaOfPentagram ( d ) ) NEW_LINE
def anglequichord ( z ) : NEW_LINE INDENT print ( " The ▁ angle ▁ is ▁ " , z , " ▁ degrees " ) NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT z = 48 NEW_LINE anglequichord ( z ) NEW_LINE DEDENT
def convertToASCII ( N ) : NEW_LINE INDENT num = str ( N ) NEW_LINE i = 0 NEW_LINE for ch in num : NEW_LINE INDENT print ( ch , " ( " , ord ( ch ) , " ) " ) NEW_LINE DEDENT DEDENT
N = 36 NEW_LINE convertToASCII ( N ) NEW_LINE
import math NEW_LINE
def productExceptSelf ( arr , N ) : NEW_LINE
product = 1 NEW_LINE
z = 0 NEW_LINE
for i in range ( N ) : NEW_LINE
if ( arr [ i ] != 0 ) : NEW_LINE INDENT product *= arr [ i ] NEW_LINE DEDENT
if ( arr [ i ] == 0 ) : NEW_LINE INDENT z += 1 NEW_LINE DEDENT
a = abs ( product ) NEW_LINE for i in range ( N ) : NEW_LINE
if ( z == 1 ) : NEW_LINE
if ( arr [ i ] != 0 ) : NEW_LINE INDENT arr [ i ] = 0 NEW_LINE DEDENT
else : NEW_LINE INDENT arr [ i ] = product NEW_LINE DEDENT continue NEW_LINE
elif ( z > 1 ) : NEW_LINE
arr [ i ] = 0 NEW_LINE continue NEW_LINE
b = abs ( arr [ i ] ) NEW_LINE
curr = round ( math . exp ( math . log ( a ) - math . log ( b ) ) ) NEW_LINE
if ( arr [ i ] < 0 and product < 0 ) : NEW_LINE INDENT arr [ i ] = curr NEW_LINE DEDENT
elif ( arr [ i ] > 0 and product > 0 ) : NEW_LINE INDENT arr [ i ] = curr NEW_LINE DEDENT
else : NEW_LINE INDENT arr [ i ] = - 1 * curr NEW_LINE DEDENT
for i in range ( N ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT
arr = [ 10 , 3 , 5 , 6 , 2 ] NEW_LINE N = len ( arr ) NEW_LINE
productExceptSelf ( arr , N ) NEW_LINE
def singleDigitSubarrayCount ( arr , N ) : NEW_LINE
res = 0 NEW_LINE
count = 0 NEW_LINE
for i in range ( N ) : NEW_LINE INDENT if ( arr [ i ] <= 9 ) : NEW_LINE DEDENT
count += 1 NEW_LINE
res += count NEW_LINE else : NEW_LINE
count = 0 NEW_LINE print ( res ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
arr = [ 0 , 1 , 14 , 2 , 5 ] NEW_LINE
N = len ( arr ) NEW_LINE singleDigitSubarrayCount ( arr , N ) NEW_LINE
def isPossible ( N ) : NEW_LINE INDENT return ( ( N & ( N - 1 ) ) and N ) NEW_LINE DEDENT
def countElements ( N ) : NEW_LINE
count = 0 NEW_LINE for i in range ( 1 , N + 1 ) : NEW_LINE INDENT if ( isPossible ( i ) ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT DEDENT print ( count ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 15 NEW_LINE countElements ( N ) NEW_LINE DEDENT
def countElements ( N ) : NEW_LINE INDENT Cur_Ele = 1 NEW_LINE Count = 0 NEW_LINE DEDENT
while ( Cur_Ele <= N ) : NEW_LINE
Count += 1 NEW_LINE
Cur_Ele = Cur_Ele * 2 NEW_LINE print ( N - Count ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 15 NEW_LINE countElements ( N ) NEW_LINE DEDENT
import sys NEW_LINE
def maxAdjacent ( arr , N ) : NEW_LINE INDENT res = [ ] NEW_LINE arr_max = - sys . maxsize - 1 NEW_LINE DEDENT
for i in range ( 1 , N ) : NEW_LINE INDENT arr_max = max ( arr_max , abs ( arr [ i - 1 ] - arr [ i ] ) ) NEW_LINE DEDENT for i in range ( 1 , N - 1 ) : NEW_LINE INDENT curr_max = abs ( arr [ i - 1 ] - arr [ i + 1 ] ) NEW_LINE DEDENT
ans = max ( curr_max , arr_max ) NEW_LINE
res . append ( ans ) NEW_LINE
for x in res : NEW_LINE INDENT print ( x , end = " ▁ " ) NEW_LINE DEDENT print ( ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 1 , 3 , 4 , 7 , 8 ] NEW_LINE N = len ( arr ) NEW_LINE maxAdjacent ( arr , N ) NEW_LINE DEDENT
def minimumIncrement ( arr , N ) : NEW_LINE
if ( N % 2 != 0 ) : NEW_LINE INDENT print ( " - 1" ) NEW_LINE return NEW_LINE DEDENT
cntEven = 0 NEW_LINE
cntOdd = 0 NEW_LINE
for i in range ( N ) : NEW_LINE
if ( arr [ i ] % 2 == 0 ) : NEW_LINE
cntEven += 1 NEW_LINE
cntOdd = N - cntEven NEW_LINE
return abs ( cntEven - cntOdd ) // 2 NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 1 , 3 , 4 , 9 ] NEW_LINE N = len ( arr ) NEW_LINE DEDENT
print ( minimumIncrement ( arr , N ) ) NEW_LINE
def cntWaysConsArray ( A , N ) : NEW_LINE
total = 1 ; NEW_LINE
oddArray = 1 ; NEW_LINE
for i in range ( N ) : NEW_LINE
total = total * 3 ; NEW_LINE
if ( A [ i ] % 2 == 0 ) : NEW_LINE
oddArray *= 2 ; NEW_LINE
print ( total - oddArray ) ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT A = [ 2 , 4 ] ; NEW_LINE N = len ( A ) ; NEW_LINE cntWaysConsArray ( A , N ) ; NEW_LINE DEDENT
def countNumberHavingKthBitSet ( N , K ) : NEW_LINE
numbers_rightmost_setbit_K = 0 NEW_LINE for i in range ( 1 , K + 1 ) : NEW_LINE
numbers_rightmost_bit_i = ( N + 1 ) // 2 NEW_LINE
N -= numbers_rightmost_bit_i NEW_LINE
if ( i == K ) : NEW_LINE INDENT numbers_rightmost_setbit_K = numbers_rightmost_bit_i NEW_LINE DEDENT print ( numbers_rightmost_setbit_K ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 15 NEW_LINE K = 2 NEW_LINE countNumberHavingKthBitSet ( N , K ) NEW_LINE DEDENT
def countSetBits ( N : int ) -> int : NEW_LINE INDENT count = 0 NEW_LINE DEDENT
while ( N ) : NEW_LINE INDENT N = N & ( N - 1 ) NEW_LINE count += 1 NEW_LINE DEDENT
return count NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT N = 4 NEW_LINE bits = countSetBits ( N ) NEW_LINE DEDENT
print ( " Odd ▁ : ▁ { } " . format ( pow ( 2 , bits ) ) ) NEW_LINE
print ( " Even ▁ : ▁ { } " . format ( N + 1 - pow ( 2 , bits ) ) ) NEW_LINE
def minMoves ( arr , N ) : NEW_LINE
odd_element_cnt = 0 ; NEW_LINE
for i in range ( N ) : NEW_LINE
if ( arr [ i ] % 2 != 0 ) : NEW_LINE INDENT odd_element_cnt += 1 ; NEW_LINE DEDENT
moves = ( odd_element_cnt ) // 2 ; NEW_LINE
if ( odd_element_cnt % 2 != 0 ) : NEW_LINE INDENT moves += 2 ; NEW_LINE DEDENT
print ( moves ) ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 5 , 6 , 3 , 7 , 20 ] ; NEW_LINE N = len ( arr ) ; NEW_LINE DEDENT
minMoves ( arr , N ) ; NEW_LINE
def minimumSubsetDifference ( N ) : NEW_LINE
blockOfSize8 = N // 8 NEW_LINE
str = " ABBABAAB " NEW_LINE
subsetDifference = 0 NEW_LINE
partition = " " NEW_LINE while blockOfSize8 != 0 : NEW_LINE INDENT partition = partition + str NEW_LINE blockOfSize8 = blockOfSize8 - 1 NEW_LINE DEDENT
A = [ ] NEW_LINE B = [ ] NEW_LINE for i in range ( N ) : NEW_LINE
if partition [ i ] == ' A ' : NEW_LINE INDENT A . append ( ( i + 1 ) * ( i + 1 ) ) NEW_LINE DEDENT
else : NEW_LINE INDENT B . append ( ( i + 1 ) * ( i + 1 ) ) NEW_LINE DEDENT
print ( subsetDifference ) NEW_LINE
for i in A : NEW_LINE INDENT print ( i , end = " ▁ " ) NEW_LINE DEDENT print ( ) NEW_LINE
for i in B : NEW_LINE INDENT print ( i , end = " ▁ " ) NEW_LINE DEDENT
N = 8 NEW_LINE
minimumSubsetDifference ( N ) NEW_LINE
from collections import defaultdict NEW_LINE
def findTheGreatestX ( P , Q ) : NEW_LINE
divisiors = defaultdict ( int ) NEW_LINE i = 2 NEW_LINE while i * i <= Q : NEW_LINE INDENT while ( Q % i == 0 and Q > 1 ) : NEW_LINE INDENT Q //= i NEW_LINE DEDENT DEDENT
divisiors [ i ] += 1 NEW_LINE i += 1 NEW_LINE
if ( Q > 1 ) : NEW_LINE INDENT divisiors [ Q ] += 1 NEW_LINE DEDENT
ans = 0 NEW_LINE
for i in divisiors : NEW_LINE INDENT frequency = divisiors [ i ] NEW_LINE temp = P NEW_LINE DEDENT
cur = 0 NEW_LINE while ( temp % i == 0 ) : NEW_LINE INDENT temp //= i NEW_LINE DEDENT
cur += 1 NEW_LINE
if ( cur < frequency ) : NEW_LINE INDENT ans = P NEW_LINE break NEW_LINE DEDENT temp = P NEW_LINE
for j in range ( cur , frequency - 1 , - 1 ) : NEW_LINE INDENT temp //= i NEW_LINE DEDENT
ans = max ( temp , ans ) NEW_LINE
print ( ans ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
P = 10 NEW_LINE Q = 4 NEW_LINE
findTheGreatestX ( P , Q ) NEW_LINE
def checkRearrangements ( mat , N , M ) : NEW_LINE
for i in range ( N ) : NEW_LINE INDENT for j in range ( 1 , M ) : NEW_LINE INDENT if ( mat [ i ] [ 0 ] != mat [ i ] [ j ] ) : NEW_LINE INDENT return " Yes " NEW_LINE DEDENT DEDENT DEDENT return " No " NEW_LINE
def nonZeroXor ( mat , N , M ) : NEW_LINE INDENT res = 0 NEW_LINE DEDENT
for i in range ( N ) : NEW_LINE INDENT res = res ^ mat [ i ] [ 0 ] NEW_LINE DEDENT
if ( res != 0 ) : NEW_LINE INDENT return " Yes " NEW_LINE DEDENT
else : NEW_LINE INDENT return checkRearrangements ( mat , N , M ) NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE
mat = [ [ 1 , 1 , 2 ] , [ 2 , 2 , 2 ] , [ 3 , 3 , 3 ] ] NEW_LINE N = len ( mat ) NEW_LINE M = len ( mat [ 0 ] ) NEW_LINE
print ( nonZeroXor ( mat , N , M ) ) NEW_LINE
def functionMax ( arr , n ) : NEW_LINE
setBit = [ [ ] for i in range ( 32 ) ] NEW_LINE for i in range ( n ) : NEW_LINE INDENT for j in range ( 32 ) : NEW_LINE DEDENT
if ( arr [ i ] & ( 1 << j ) ) : NEW_LINE
setBit [ j ] . append ( i ) NEW_LINE
i = 31 NEW_LINE while ( i >= 0 ) : NEW_LINE INDENT if ( len ( setBit [ i ] ) == 1 ) : NEW_LINE DEDENT
temp = arr [ 0 ] NEW_LINE arr [ 0 ] = arr [ setBit [ i ] [ 0 ] ] NEW_LINE arr [ setBit [ i ] [ 0 ] ] = temp NEW_LINE break NEW_LINE i -= 1 NEW_LINE
maxAnd = arr [ 0 ] NEW_LINE for i in range ( 1 , n , 1 ) : NEW_LINE INDENT maxAnd = ( maxAnd & ( ~ arr [ i ] ) ) NEW_LINE DEDENT
return maxAnd NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 1 , 2 , 4 , 8 , 16 ] NEW_LINE n = len ( arr ) NEW_LINE DEDENT
print ( functionMax ( arr , n ) ) NEW_LINE
def nCr ( n , r ) : NEW_LINE
res = 1 NEW_LINE
if r > n - r : NEW_LINE INDENT r = n - r NEW_LINE DEDENT
for i in range ( r ) : NEW_LINE INDENT res *= ( n - i ) NEW_LINE res /= ( i + 1 ) NEW_LINE DEDENT return res ; NEW_LINE
def solve ( n , m , k ) : NEW_LINE
sum = 0 ; NEW_LINE
for i in range ( k + 1 ) : NEW_LINE INDENT sum += nCr ( n , i ) * nCr ( m , k - i ) NEW_LINE DEDENT return int ( sum ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 3 NEW_LINE m = 2 NEW_LINE k = 2 ; NEW_LINE print ( solve ( n , m , k ) ) NEW_LINE DEDENT
def powerOptimised ( a , n ) : NEW_LINE
ans = 1 NEW_LINE while ( n > 0 ) : NEW_LINE INDENT last_bit = ( n & 1 ) NEW_LINE DEDENT
if ( last_bit ) : NEW_LINE INDENT ans = ans * a NEW_LINE DEDENT a = a * a NEW_LINE
n = n >> 1 NEW_LINE return ans NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT a = 3 NEW_LINE n = 5 NEW_LINE print ( powerOptimised ( a , n ) ) NEW_LINE DEDENT
def findMaximumGcd ( n ) : NEW_LINE
max_gcd = 1 NEW_LINE i = 1 NEW_LINE
while ( i * i <= n ) : NEW_LINE
if n % i == 0 : NEW_LINE
if ( i > max_gcd ) : NEW_LINE INDENT max_gcd = i NEW_LINE DEDENT if ( ( n / i != i ) and ( n / i != n ) and ( ( n / i ) > max_gcd ) ) : NEW_LINE INDENT max_gcd = n / i NEW_LINE DEDENT i += 1 NEW_LINE
return ( int ( max_gcd ) ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
n = 10 NEW_LINE
print ( findMaximumGcd ( n ) ) NEW_LINE
x = 2000021 NEW_LINE
v = [ 0 ] * x NEW_LINE
def sieve ( ) : NEW_LINE INDENT v [ 1 ] = 1 NEW_LINE DEDENT
for i in range ( 2 , x ) : NEW_LINE INDENT v [ i ] = i NEW_LINE DEDENT
for i in range ( 4 , x , 2 ) : NEW_LINE INDENT v [ i ] = 2 NEW_LINE DEDENT i = 3 NEW_LINE while ( i * i < x ) : NEW_LINE
if ( v [ i ] == i ) : NEW_LINE
for j in range ( i * i , x , i ) : NEW_LINE
if ( v [ j ] == j ) : NEW_LINE INDENT v [ j ] = i NEW_LINE DEDENT i += 1 NEW_LINE
def prime_factors ( n ) : NEW_LINE INDENT s = set ( ) NEW_LINE while ( n != 1 ) : NEW_LINE INDENT s . add ( v [ n ] ) NEW_LINE n = n // v [ n ] NEW_LINE DEDENT return len ( s ) NEW_LINE DEDENT
def distinctPrimes ( m , k ) : NEW_LINE
result = [ ] NEW_LINE for i in range ( 14 , m + k ) : NEW_LINE
count = prime_factors ( i ) NEW_LINE
if ( count == k ) : NEW_LINE INDENT result . append ( i ) NEW_LINE DEDENT p = len ( result ) NEW_LINE for index in range ( p - 1 ) : NEW_LINE element = result [ index ] NEW_LINE count = 1 NEW_LINE z = index NEW_LINE
while ( z < p - 1 and count <= k and result [ z ] + 1 == result [ z + 1 ] ) : NEW_LINE
count += 1 NEW_LINE z += 1 NEW_LINE
if ( count >= k ) : NEW_LINE INDENT print ( element , end = ' ▁ ' ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE
sieve ( ) NEW_LINE
N = 1000 NEW_LINE K = 3 NEW_LINE
distinctPrimes ( N , K ) NEW_LINE
def print_product ( a , b , c , d ) : NEW_LINE
prod1 = a * c NEW_LINE prod2 = b * d NEW_LINE prod3 = ( a + b ) * ( c + d ) NEW_LINE
real = prod1 - prod2 NEW_LINE
imag = prod3 - ( prod1 + prod2 ) NEW_LINE
print ( real , " ▁ + ▁ " , imag , " i " ) NEW_LINE
a = 2 NEW_LINE b = 3 NEW_LINE c = 4 NEW_LINE d = 5 NEW_LINE
print_product ( a , b , c , d ) NEW_LINE
def isInsolite ( n ) : NEW_LINE INDENT N = n ; NEW_LINE DEDENT
sum = 0 ; NEW_LINE
product = 1 ; NEW_LINE while ( n != 0 ) : NEW_LINE
r = n % 10 ; NEW_LINE sum = sum + r * r ; NEW_LINE product = product * r * r ; NEW_LINE n = n // 10 ; NEW_LINE return ( ( N % sum == 0 ) and ( N % product == 0 ) ) ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 111 ; NEW_LINE DEDENT
if ( isInsolite ( N ) ) : NEW_LINE INDENT print ( " Yes " ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) ; NEW_LINE DEDENT
def sigma ( n ) : NEW_LINE INDENT if ( n == 1 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT DEDENT
result = 0 NEW_LINE
' NEW_LINE INDENT for i in range ( 2 , pow ( n , 1 // 2 ) ) : NEW_LINE DEDENT
' NEW_LINE INDENT if ( n % i == 0 ) : NEW_LINE DEDENT
if ( i == ( n / i ) ) : NEW_LINE INDENT result += i NEW_LINE DEDENT else : NEW_LINE INDENT result += ( i + n / i ) NEW_LINE DEDENT
return ( result + n + 1 ) NEW_LINE
def isSuperabundant ( N ) : NEW_LINE
for i in range ( 1 , N ) : NEW_LINE INDENT x = sigma ( ( int ) ( i ) ) / i NEW_LINE y = sigma ( ( int ) ( N ) ) / ( N * 1.0 ) NEW_LINE if ( x > y ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT return True NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 4 NEW_LINE if ( isSuperabundant ( N ) != True ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT DEDENT
import math NEW_LINE
def isDNum ( n ) : NEW_LINE
if n < 4 : NEW_LINE INDENT return False NEW_LINE DEDENT
for k in range ( 2 , n ) : NEW_LINE INDENT numerator = pow ( k , n - 2 ) - k NEW_LINE hcf = math . gcd ( n , k ) NEW_LINE DEDENT
if ( hcf == 1 and ( numerator % n ) != 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT return True NEW_LINE
n = 15 NEW_LINE if isDNum ( n ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
def Sum ( N ) : NEW_LINE INDENT SumOfPrimeDivisors = [ 0 ] * ( N + 1 ) NEW_LINE for i in range ( 2 , N + 1 ) : NEW_LINE DEDENT
if ( SumOfPrimeDivisors [ i ] == 0 ) : NEW_LINE
for j in range ( i , N + 1 , i ) : NEW_LINE INDENT SumOfPrimeDivisors [ j ] += i NEW_LINE DEDENT return SumOfPrimeDivisors [ N ] NEW_LINE
def RuthAaronNumber ( n ) : NEW_LINE INDENT if ( Sum ( n ) == Sum ( n + 1 ) ) : NEW_LINE INDENT return True NEW_LINE DEDENT else : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT
N = 714 NEW_LINE if ( RuthAaronNumber ( N ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
def maxAdjacentDifference ( N , K ) : NEW_LINE
if ( N == 1 ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT
if ( N == 2 ) : NEW_LINE INDENT return K ; NEW_LINE DEDENT
return 2 * K ; NEW_LINE
N = 6 ; NEW_LINE K = 11 ; NEW_LINE print ( maxAdjacentDifference ( N , K ) ) ; NEW_LINE
mod = 1000000007 NEW_LINE
def linearSum ( n ) : NEW_LINE INDENT return n * ( n + 1 ) // 2 % mod NEW_LINE DEDENT
def rangeSum ( b , a ) : NEW_LINE INDENT return ( linearSum ( b ) - ( linearSum ( a ) ) ) % mod NEW_LINE DEDENT
def totalSum ( n ) : NEW_LINE
result = 0 NEW_LINE i = 1 NEW_LINE
while True : NEW_LINE
result += rangeSum ( n // i , n // ( i + 1 ) ) * ( i % mod ) % mod ; NEW_LINE result %= mod ; NEW_LINE if i == n : NEW_LINE INDENT break NEW_LINE DEDENT i = n // ( n // ( i + 1 ) ) NEW_LINE return result NEW_LINE
N = 4 NEW_LINE print ( totalSum ( N ) ) NEW_LINE N = 12 NEW_LINE print ( totalSum ( N ) ) NEW_LINE
def isDouble ( num ) : NEW_LINE INDENT s = str ( num ) NEW_LINE l = len ( s ) NEW_LINE DEDENT
if ( s [ 0 ] == s [ 1 ] ) : NEW_LINE INDENT return False NEW_LINE DEDENT
if ( l % 2 == 1 ) : NEW_LINE INDENT s = s + s [ 1 ] NEW_LINE l += 1 NEW_LINE DEDENT
s1 = s [ : l // 2 ] NEW_LINE
s2 = s [ l // 2 : ] NEW_LINE
return s1 == s2 NEW_LINE
def isNontrivialUndulant ( N ) : NEW_LINE INDENT return N > 100 and isDouble ( N ) NEW_LINE DEDENT
n = 121 NEW_LINE if ( isNontrivialUndulant ( n ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
def MegagonNum ( n ) : NEW_LINE INDENT return ( 999998 * n * n - 999996 * n ) // 2 ; NEW_LINE DEDENT
n = 3 ; NEW_LINE print ( MegagonNum ( n ) ) ; NEW_LINE
mod = 1000000007 ; NEW_LINE
def productPairs ( arr , n ) : NEW_LINE
product = 1 ; NEW_LINE
for i in range ( n ) : NEW_LINE INDENT for j in range ( n ) : NEW_LINE DEDENT
product *= ( arr [ i ] % mod * arr [ j ] % mod ) % mod ; NEW_LINE product = product % mod ; NEW_LINE
return product % mod ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 1 , 2 , 3 ] ; NEW_LINE n = len ( arr ) ; NEW_LINE print ( productPairs ( arr , n ) ) ; NEW_LINE DEDENT
mod = 1000000007 NEW_LINE
def power ( x , y ) : NEW_LINE INDENT p = 1000000007 NEW_LINE DEDENT
res = 1 NEW_LINE
x = x % p NEW_LINE while ( y > 0 ) : NEW_LINE
if ( ( y & 1 ) != 0 ) : NEW_LINE INDENT res = ( res * x ) % p NEW_LINE DEDENT y = y >> 1 NEW_LINE x = ( x * x ) % p NEW_LINE
return res NEW_LINE
def productPairs ( arr , n ) : NEW_LINE
product = 1 NEW_LINE
for i in range ( n ) : NEW_LINE
product = ( product % mod * ( int ) ( power ( arr [ i ] , ( 2 * n ) ) ) % mod ) % mod NEW_LINE return ( product % mod ) NEW_LINE
arr = [ 1 , 2 , 3 ] NEW_LINE n = len ( arr ) NEW_LINE print ( productPairs ( arr , n ) ) NEW_LINE
def constructArray ( N ) : NEW_LINE INDENT arr = [ 0 ] * N NEW_LINE DEDENT
for i in range ( 1 , N + 1 ) : NEW_LINE INDENT arr [ i - 1 ] = i ; NEW_LINE DEDENT
for i in range ( N ) : NEW_LINE INDENT print ( arr [ i ] , end = " , ▁ " ) NEW_LINE DEDENT
N = 6 ; NEW_LINE constructArray ( N ) ; NEW_LINE
def isPrime ( n ) : NEW_LINE INDENT if ( n <= 1 ) : NEW_LINE INDENT return False ; NEW_LINE DEDENT for i in range ( 2 , n ) : NEW_LINE INDENT if ( n % i == 0 ) : NEW_LINE INDENT return False ; NEW_LINE DEDENT DEDENT return True ; NEW_LINE DEDENT
def countSubsequences ( arr , n ) : NEW_LINE
totalSubsequence = ( int ) ( pow ( 2 , n ) - 1 ) ; NEW_LINE countPrime = 0 ; NEW_LINE countOnes = 0 ; NEW_LINE
for i in range ( n ) : NEW_LINE INDENT if ( arr [ i ] == 1 ) : NEW_LINE INDENT countOnes += 1 ; NEW_LINE DEDENT elif ( isPrime ( arr [ i ] ) ) : NEW_LINE INDENT countPrime += 1 ; NEW_LINE DEDENT DEDENT compositeSubsequence = 0 ; NEW_LINE
onesSequence = ( int ) ( pow ( 2 , countOnes ) - 1 ) ; NEW_LINE
compositeSubsequence = ( totalSubsequence - countPrime - onesSequence - onesSequence * countPrime ) ; NEW_LINE return compositeSubsequence ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 2 , 1 , 2 ] ; NEW_LINE n = len ( arr ) ; NEW_LINE print ( countSubsequences ( arr , n ) ) ; NEW_LINE DEDENT
def checksum ( n , k ) : NEW_LINE
first_term = ( ( 2 * n ) / k + ( 1 - k ) ) / 2.0 NEW_LINE
if ( first_term - int ( first_term ) == 0 ) : NEW_LINE
for i in range ( int ( first_term ) , int ( first_term ) + k ) : NEW_LINE INDENT print ( i , end = ' ▁ ' ) NEW_LINE DEDENT else : NEW_LINE print ( ' - 1' ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT ( n , k ) = ( 33 , 6 ) NEW_LINE checksum ( n , k ) NEW_LINE DEDENT
def sumEvenNumbers ( N , K ) : NEW_LINE INDENT check = N - 2 * ( K - 1 ) NEW_LINE DEDENT
if ( check > 0 and check % 2 == 0 ) : NEW_LINE INDENT for i in range ( K - 1 ) : NEW_LINE INDENT print ( "2 ▁ " , end = " " ) NEW_LINE DEDENT print ( check ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " - 1" ) NEW_LINE DEDENT
N = 8 NEW_LINE K = 2 NEW_LINE sumEvenNumbers ( N , K ) NEW_LINE
def calculateWays ( N ) : NEW_LINE INDENT x = 0 ; NEW_LINE v = [ ] ; NEW_LINE DEDENT
for i in range ( N ) : NEW_LINE INDENT v . append ( 0 ) ; NEW_LINE DEDENT
for i in range ( N // 2 + 1 ) : NEW_LINE
if ( N % 2 == 0 and i == N // 2 ) : NEW_LINE INDENT break ; NEW_LINE DEDENT
x = N * ( i + 1 ) - ( i + 1 ) * i ; NEW_LINE
v [ i ] = x ; NEW_LINE v [ N - i - 1 ] = x ; NEW_LINE return v ; NEW_LINE
def printArray ( v ) : NEW_LINE INDENT for i in range ( len ( v ) ) : NEW_LINE INDENT print ( v [ i ] , end = " ▁ " ) ; NEW_LINE DEDENT DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT v = calculateWays ( 4 ) ; NEW_LINE printArray ( v ) ; NEW_LINE DEDENT
MAXN = 10000000 NEW_LINE
def sumOfDigits ( n ) : NEW_LINE
sum = 0 NEW_LINE while ( n > 0 ) : NEW_LINE
sum += n % 10 NEW_LINE
n //= 10 NEW_LINE return sum NEW_LINE
def smallestNum ( X , Y ) : NEW_LINE
res = - 1 ; NEW_LINE
for i in range ( X , MAXN ) : NEW_LINE
sum_of_digit = sumOfDigits ( i ) NEW_LINE
if sum_of_digit % Y == 0 : NEW_LINE INDENT res = i NEW_LINE break NEW_LINE DEDENT return res NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT ( X , Y ) = ( 5923 , 13 ) NEW_LINE print ( smallestNum ( X , Y ) ) NEW_LINE DEDENT
def countValues ( N ) : NEW_LINE INDENT div = [ ] NEW_LINE i = 2 NEW_LINE DEDENT
while ( ( i * i ) <= N ) : NEW_LINE
if ( N % i == 0 ) : NEW_LINE INDENT div . append ( i ) NEW_LINE DEDENT
if ( N != i * i ) : NEW_LINE INDENT div . append ( N // i ) NEW_LINE DEDENT i += 1 NEW_LINE answer = 0 NEW_LINE i = 1 NEW_LINE
while ( ( i * i ) <= N - 1 ) : NEW_LINE
if ( ( N - 1 ) % i == 0 ) : NEW_LINE INDENT if ( i * i == N - 1 ) : NEW_LINE INDENT answer += 1 NEW_LINE DEDENT else : NEW_LINE INDENT answer += 2 NEW_LINE DEDENT DEDENT i += 1 NEW_LINE
for d in div : NEW_LINE INDENT K = N NEW_LINE while ( K % d == 0 ) : NEW_LINE INDENT K //= d NEW_LINE DEDENT if ( ( K - 1 ) % d == 0 ) : NEW_LINE INDENT answer += 1 NEW_LINE DEDENT DEDENT return answer NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT N = 6 NEW_LINE print ( countValues ( N ) ) NEW_LINE DEDENT
def findMaxPrimeDivisor ( n ) : NEW_LINE INDENT max_possible_prime = 0 NEW_LINE DEDENT
while ( n % 2 == 0 ) : NEW_LINE INDENT max_possible_prime += 1 NEW_LINE n = n // 2 NEW_LINE DEDENT
i = 3 NEW_LINE while ( i * i <= n ) : NEW_LINE INDENT while ( n % i == 0 ) : NEW_LINE INDENT max_possible_prime += 1 NEW_LINE n = n // i NEW_LINE DEDENT i = i + 2 NEW_LINE DEDENT
if ( n > 2 ) : NEW_LINE INDENT max_possible_prime += 1 NEW_LINE DEDENT print ( max_possible_prime ) NEW_LINE
n = 4 NEW_LINE
findMaxPrimeDivisor ( n ) NEW_LINE
def CountWays ( n ) : NEW_LINE INDENT ans = ( n - 1 ) // 2 NEW_LINE return ans NEW_LINE DEDENT
N = 8 NEW_LINE print ( CountWays ( N ) ) NEW_LINE
def Solve ( arr , size , n ) : NEW_LINE INDENT v = [ 0 ] * ( n + 1 ) ; NEW_LINE DEDENT
for i in range ( size ) : NEW_LINE INDENT v [ arr [ i ] ] += 1 NEW_LINE DEDENT
max1 = max ( set ( arr ) , key = v . count ) NEW_LINE
diff1 = n + 1 - v . count ( 0 ) NEW_LINE
max_size = max ( min ( v [ max1 ] - 1 , diff1 ) , min ( v [ max1 ] , diff1 - 1 ) ) NEW_LINE print ( " Maximum ▁ size ▁ is ▁ : " , max_size ) NEW_LINE
print ( " The ▁ First ▁ Array ▁ Is ▁ : ▁ " ) NEW_LINE for i in range ( max_size ) : NEW_LINE INDENT print ( max1 , end = " ▁ " ) NEW_LINE v [ max1 ] -= 1 NEW_LINE DEDENT print ( ) NEW_LINE
print ( " The ▁ Second ▁ Array ▁ Is ▁ : ▁ " ) NEW_LINE for i in range ( n + 1 ) : NEW_LINE INDENT if ( v [ i ] > 0 ) : NEW_LINE INDENT print ( i , end = " ▁ " ) NEW_LINE max_size -= 1 NEW_LINE DEDENT if ( max_size < 1 ) : NEW_LINE INDENT break NEW_LINE DEDENT DEDENT print ( ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
n = 7 NEW_LINE
arr = [ 1 , 2 , 1 , 5 , 1 , 6 , 7 , 2 ] NEW_LINE
size = len ( arr ) NEW_LINE Solve ( arr , size , n ) NEW_LINE
def power ( x , y , p ) : NEW_LINE
res = 1 NEW_LINE
x = x % p NEW_LINE while ( y > 0 ) : NEW_LINE
if ( y & 1 ) : NEW_LINE INDENT res = ( res * x ) % p NEW_LINE DEDENT
x = ( x * x ) % p NEW_LINE return res NEW_LINE
def modInverse ( n , p ) : NEW_LINE INDENT return power ( n , p - 2 , p ) NEW_LINE DEDENT
def nCrModPFermat ( n , r , p ) : NEW_LINE
if ( r == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT if ( n < r ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
fac = [ 0 ] * ( n + 1 ) NEW_LINE fac [ 0 ] = 1 NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE INDENT fac [ i ] = fac [ i - 1 ] * i % p NEW_LINE DEDENT return ( fac [ n ] * modInverse ( fac [ r ] , p ) % p * modInverse ( fac [ n - r ] , p ) % p ) % p NEW_LINE
def SumOfXor ( a , n ) : NEW_LINE INDENT mod = 10037 NEW_LINE answer = 0 NEW_LINE DEDENT
for k in range ( 32 ) : NEW_LINE
x = 0 NEW_LINE y = 0 NEW_LINE for i in range ( n ) : NEW_LINE
if ( a [ i ] & ( 1 << k ) ) : NEW_LINE INDENT x += 1 NEW_LINE DEDENT else : NEW_LINE INDENT y += 1 NEW_LINE DEDENT
answer += ( ( 1 << k ) % mod * ( nCrModPFermat ( x , 3 , mod ) + x * nCrModPFermat ( y , 2 , mod ) ) % mod ) % mod NEW_LINE return answer NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 5 NEW_LINE A = [ 3 , 5 , 2 , 18 , 7 ] NEW_LINE print ( SumOfXor ( A , n ) ) NEW_LINE DEDENT
import math NEW_LINE
def probability ( N ) : NEW_LINE
a = 2 NEW_LINE b = 3 NEW_LINE
if N == 1 : NEW_LINE INDENT return a NEW_LINE DEDENT elif N == 2 : NEW_LINE INDENT return b NEW_LINE DEDENT else : NEW_LINE
for i in range ( 3 , N + 1 ) : NEW_LINE INDENT c = a + b NEW_LINE a = b NEW_LINE b = c NEW_LINE DEDENT return b NEW_LINE
def operations ( N ) : NEW_LINE
x = probability ( N ) NEW_LINE
y = math . pow ( 2 , N ) NEW_LINE return round ( x / y , 2 ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 10 NEW_LINE print ( operations ( N ) ) NEW_LINE DEDENT
def isPerfectCube ( x ) : NEW_LINE INDENT x = abs ( x ) NEW_LINE return int ( round ( x ** ( 1. / 3 ) ) ) ** 3 == x NEW_LINE DEDENT
def checkCube ( a , b ) : NEW_LINE
s1 = str ( a ) NEW_LINE s2 = str ( b ) NEW_LINE
c = int ( s1 + s2 ) NEW_LINE
if ( isPerfectCube ( c ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT a = 6 NEW_LINE b = 4 NEW_LINE checkCube ( a , b ) NEW_LINE DEDENT
def largest_sum ( arr , n ) : NEW_LINE
maximum = - 1 NEW_LINE
m = dict ( ) NEW_LINE
for i in arr : NEW_LINE INDENT m [ i ] = m . get ( i , 0 ) + 1 NEW_LINE DEDENT
for j in list ( m ) : NEW_LINE
if ( ( j in m ) and m [ j ] > 1 ) : NEW_LINE
x , y = 0 , 0 NEW_LINE if 2 * j in m : NEW_LINE INDENT m [ 2 * j ] = m [ 2 * j ] + m [ j ] // 2 NEW_LINE DEDENT else : NEW_LINE INDENT m [ 2 * j ] = m [ j ] // 2 NEW_LINE DEDENT
if ( 2 * j > maximum ) : NEW_LINE INDENT maximum = 2 * j NEW_LINE DEDENT
return maximum NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 1 , 1 , 2 , 4 , 7 , 8 ] NEW_LINE n = len ( arr ) NEW_LINE DEDENT
print ( largest_sum ( arr , n ) ) NEW_LINE
def canBeReduced ( x , y ) : NEW_LINE INDENT maxi = max ( x , y ) NEW_LINE mini = min ( x , y ) NEW_LINE DEDENT
if ( ( ( x + y ) % 3 ) == 0 and maxi <= 2 * mini ) : NEW_LINE INDENT print ( " YES " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NO " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT x = 6 NEW_LINE y = 9 NEW_LINE DEDENT
canBeReduced ( x , y ) NEW_LINE
import math NEW_LINE
def isPrime ( N ) : NEW_LINE INDENT isPrime = True ; NEW_LINE DEDENT
arr = [ 7 , 11 , 13 , 17 , 19 , 23 , 29 , 31 ] NEW_LINE
if ( N < 2 ) : NEW_LINE INDENT isPrime = False NEW_LINE DEDENT
if ( N % 2 == 0 or N % 3 == 0 or N % 5 == 0 ) : NEW_LINE INDENT isPrime = False NEW_LINE DEDENT
for i in range ( 0 , int ( math . sqrt ( N ) ) , 30 ) : NEW_LINE
for c in arr : NEW_LINE
if ( c > int ( math . sqrt ( N ) ) ) : NEW_LINE INDENT break NEW_LINE DEDENT
else : NEW_LINE INDENT if ( N % ( c + i ) == 0 ) : NEW_LINE INDENT isPrime = False NEW_LINE break NEW_LINE DEDENT DEDENT
if ( not isPrime ) : NEW_LINE INDENT break NEW_LINE DEDENT if ( isPrime ) : NEW_LINE print ( " Prime ▁ Number " ) NEW_LINE else : NEW_LINE print ( " Not ▁ a ▁ Prime ▁ Number " ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT N = 121 NEW_LINE DEDENT
isPrime ( N ) NEW_LINE
def printPairs ( arr , n ) : NEW_LINE
for i in range ( n ) : NEW_LINE INDENT for j in range ( n ) : NEW_LINE INDENT print ( " ( " , arr [ i ] , " , " , arr [ j ] , " ) " , end = " , ▁ " ) NEW_LINE DEDENT DEDENT
arr = [ 1 , 2 ] NEW_LINE n = len ( arr ) NEW_LINE printPairs ( arr , n ) NEW_LINE
def circle ( x1 , y1 , x2 , y2 , r1 , r2 ) : NEW_LINE INDENT distSq = ( ( ( x1 - x2 ) * ( x1 - x2 ) ) + ( ( y1 - y2 ) * ( y1 - y2 ) ) ) ** ( .5 ) NEW_LINE if ( distSq + r2 == r1 ) : NEW_LINE INDENT print ( " The ▁ smaller ▁ circle ▁ lies ▁ completely " " ▁ inside ▁ the ▁ bigger ▁ circle ▁ with ▁ " " touching ▁ each ▁ other ▁ " " at ▁ a ▁ point ▁ of ▁ circumference . ▁ " ) NEW_LINE DEDENT elif ( distSq + r2 < r1 ) : NEW_LINE INDENT print ( " The ▁ smaller ▁ circle ▁ lies ▁ completely " " ▁ inside ▁ the ▁ bigger ▁ circle ▁ without " " ▁ touching ▁ each ▁ other ▁ " " at ▁ a ▁ point ▁ of ▁ circumference . ▁ " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " The ▁ smaller ▁ does ▁ not ▁ lies ▁ inside " " ▁ the ▁ bigger ▁ circle ▁ completely . " ) NEW_LINE DEDENT DEDENT
x1 , y1 = 10 , 8 NEW_LINE x2 , y2 = 1 , 2 NEW_LINE r1 , r2 = 30 , 10 NEW_LINE circle ( x1 , y1 , x2 , y2 , r1 , r2 ) NEW_LINE
def lengtang ( r1 , r2 , d ) : NEW_LINE INDENT print ( " The ▁ length ▁ of ▁ the ▁ direct ▁ common ▁ tangent ▁ is ▁ " , ( ( d ** 2 ) - ( ( r1 - r2 ) ** 2 ) ) ** ( 1 / 2 ) ) ; NEW_LINE DEDENT
r1 = 4 ; r2 = 6 ; d = 3 ; NEW_LINE lengtang ( r1 , r2 , d ) ; NEW_LINE
def rad ( d , h ) : NEW_LINE INDENT print ( " The ▁ radius ▁ of ▁ the ▁ circle ▁ is " , ( ( d * d ) / ( 8 * h ) + h / 2 ) ) NEW_LINE DEDENT
d = 4 ; h = 1 ; NEW_LINE rad ( d , h ) ; NEW_LINE
def shortdis ( r , d ) : NEW_LINE INDENT print ( " The ▁ shortest ▁ distance ▁ " , end = " " ) ; NEW_LINE print ( " from ▁ the ▁ chord ▁ to ▁ centre ▁ " , end = " " ) ; NEW_LINE print ( ( ( r * r ) - ( ( d * d ) / 4 ) ) ** ( 1 / 2 ) ) ; NEW_LINE DEDENT
r = 4 ; NEW_LINE d = 3 ; NEW_LINE shortdis ( r , d ) ; NEW_LINE
import math NEW_LINE
def lengtang ( r1 , r2 , d ) : NEW_LINE INDENT print ( " The ▁ length ▁ of ▁ the ▁ direct ▁ common ▁ tangent ▁ is " , ( ( ( d ** 2 ) - ( ( r1 - r2 ) ** 2 ) ) ** ( 1 / 2 ) ) ) ; NEW_LINE DEDENT
r1 = 4 ; r2 = 6 ; d = 12 ; NEW_LINE lengtang ( r1 , r2 , d ) ; NEW_LINE
def square ( a ) : NEW_LINE
if ( a < 0 ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
x = 0.464 * a NEW_LINE return x NEW_LINE
a = 5 NEW_LINE print ( square ( a ) ) NEW_LINE
from math import tan NEW_LINE
def polyapothem ( n , a ) : NEW_LINE
if ( a < 0 and n < 0 ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
return a / ( 2 * tan ( ( 180 / n ) * 3.14159 / 180 ) ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT a = 9 NEW_LINE n = 6 NEW_LINE print ( ' { 0 : . 6 } ' . format ( polyapothem ( n , a ) ) ) NEW_LINE DEDENT
from math import tan NEW_LINE
def polyarea ( n , a ) : NEW_LINE
if ( a < 0 and n < 0 ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
A = ( a * a * n ) / ( 4 * tan ( ( 180 / n ) * 3.14159 / 180 ) ) NEW_LINE return A NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT a = 9 NEW_LINE n = 6 NEW_LINE print ( ' { 0 : . 6 } ' . format ( polyarea ( n , a ) ) ) NEW_LINE DEDENT
from math import sin NEW_LINE
def calculateSide ( n , r ) : NEW_LINE INDENT theta = 360 / n NEW_LINE theta_in_radians = theta * 3.14 / 180 NEW_LINE return 2 * r * sin ( theta_in_radians / 2 ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE
n = 3 NEW_LINE
r = 5 NEW_LINE print ( ' { 0 : . 5 } ' . format ( calculateSide ( n , r ) ) ) NEW_LINE
def cyl ( r , R , h ) : NEW_LINE
if ( h < 0 and r < 0 and R < 0 ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
r1 = r NEW_LINE
h1 = h NEW_LINE
V = 3.14 * pow ( r1 , 2 ) * h1 NEW_LINE return round ( V , 2 ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT r , R , h = 7 , 11 , 6 NEW_LINE print ( cyl ( r , R , h ) ) NEW_LINE DEDENT
def Perimeter ( s , n ) : NEW_LINE INDENT perimeter = 1 NEW_LINE DEDENT
perimeter = n * s NEW_LINE return perimeter NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
n = 5 NEW_LINE
s = 2.5 NEW_LINE
peri = Perimeter ( s , n ) NEW_LINE print ( " Perimeter ▁ of ▁ Regular ▁ Polygon ▁ with " , n , " sides ▁ of ▁ length " , s , " = " , peri ) NEW_LINE
def rhombusarea ( l , b ) : NEW_LINE
if ( l < 0 or b < 0 ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
return ( l * b ) / 2 NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT l = 16 NEW_LINE b = 6 NEW_LINE print ( rhombusarea ( l , b ) ) NEW_LINE DEDENT
def FindPoint ( x1 , y1 , x2 , y2 , x , y ) : NEW_LINE INDENT if ( x > x1 and x < x2 and y > y1 and y < y2 ) : NEW_LINE INDENT return True NEW_LINE DEDENT else : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE
x1 , y1 , x2 , y2 = 0 , 0 , 10 , 8 NEW_LINE
x , y = 1 , 5 NEW_LINE
if FindPoint ( x1 , y1 , x2 , y2 , x , y ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
import math NEW_LINE
def shortest_distance ( x1 , y1 , z1 , a , b , c , d ) : NEW_LINE INDENT d = abs ( ( a * x1 + b * y1 + c * z1 + d ) ) NEW_LINE e = ( math . sqrt ( a * a + b * b + c * c ) ) NEW_LINE print ( " Perpendicular ▁ distance ▁ is " , d / e ) NEW_LINE DEDENT
x1 = 4 NEW_LINE y1 = - 4 NEW_LINE z1 = 3 NEW_LINE a = 2 NEW_LINE b = - 2 NEW_LINE c = 5 NEW_LINE d = 8 NEW_LINE
shortest_distance ( x1 , y1 , z1 , a , b , c , d ) NEW_LINE
def findVolume ( l , b , h ) : NEW_LINE
return ( ( l * b * h ) / 2 ) NEW_LINE
l = 18 NEW_LINE b = 12 NEW_LINE h = 9 NEW_LINE
print ( " Volume ▁ of ▁ triangular ▁ prism : ▁ " , findVolume ( l , b , h ) ) NEW_LINE
def isRectangle ( a , b , c , d ) : NEW_LINE
if ( a == b and d == c ) or ( a == c and b == d ) or ( a == d and b == c ) : NEW_LINE INDENT return True NEW_LINE DEDENT else : NEW_LINE INDENT return False NEW_LINE DEDENT
a , b , c , d = 1 , 2 , 3 , 4 NEW_LINE print ( " Yes " if isRectangle ( a , b , c , d ) else " No " ) NEW_LINE
def midpoint ( x1 , x2 , y1 , y2 ) : NEW_LINE INDENT print ( ( x1 + x2 ) // 2 , " ▁ , ▁ " , ( y1 + y2 ) // 2 ) NEW_LINE DEDENT
x1 , y1 , x2 , y2 = - 1 , 2 , 3 , - 6 NEW_LINE midpoint ( x1 , x2 , y1 , y2 ) NEW_LINE
import math NEW_LINE
def arcLength ( diameter , angle ) : NEW_LINE INDENT if angle >= 360 : NEW_LINE INDENT print ( " Angle ▁ cannot ▁ be ▁ formed " ) NEW_LINE return 0 NEW_LINE DEDENT else : NEW_LINE INDENT arc = ( 3.142857142857143 * diameter ) * ( angle / 360.0 ) NEW_LINE return arc NEW_LINE DEDENT DEDENT
diameter = 25.0 NEW_LINE angle = 45.0 NEW_LINE arc_len = arcLength ( diameter , angle ) NEW_LINE print ( arc_len ) NEW_LINE
import math NEW_LINE def checkCollision ( a , b , c , x , y , radius ) : NEW_LINE
dist = ( ( abs ( a * x + b * y + c ) ) / math . sqrt ( a * a + b * b ) ) NEW_LINE
if ( radius == dist ) : NEW_LINE INDENT print ( " Touch " ) NEW_LINE DEDENT elif ( radius > dist ) : NEW_LINE INDENT print ( " Intersect " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Outside " ) NEW_LINE DEDENT
radius = 5 NEW_LINE x = 0 NEW_LINE y = 0 NEW_LINE a = 3 NEW_LINE b = 4 NEW_LINE c = 25 NEW_LINE checkCollision ( a , b , c , x , y , radius ) NEW_LINE
def polygonArea ( X , Y , n ) : NEW_LINE
area = 0.0 NEW_LINE
j = n - 1 NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT area = area + ( X [ j ] + X [ i ] ) * ( Y [ j ] - Y [ i ] ) NEW_LINE DEDENT
return abs ( area // 2.0 ) NEW_LINE
X = [ 0 , 2 , 4 ] NEW_LINE Y = [ 1 , 3 , 7 ] NEW_LINE n = len ( X ) NEW_LINE print ( polygonArea ( X , Y , n ) ) NEW_LINE
def chk ( v ) : NEW_LINE
v = list ( bin ( v ) [ 2 : ] ) NEW_LINE v . reverse ( ) NEW_LINE if ( '1' in v ) : NEW_LINE INDENT v = v . index ( '1' ) NEW_LINE return ( 2 ** v ) NEW_LINE DEDENT else : NEW_LINE INDENT return 0 NEW_LINE DEDENT
def sumOfLSB ( arr , N ) : NEW_LINE
lsb_arr = [ ] NEW_LINE for i in range ( N ) : NEW_LINE
lsb_arr . append ( chk ( arr [ i ] ) ) NEW_LINE
lsb_arr . sort ( reverse = True ) NEW_LINE ans = 0 NEW_LINE for i in range ( 0 , N - 1 , 2 ) : NEW_LINE
ans += ( lsb_arr [ i + 1 ] ) NEW_LINE
print ( ans ) NEW_LINE
N = 5 NEW_LINE arr = [ 1 , 2 , 3 , 4 , 5 ] NEW_LINE
sumOfLSB ( arr , N ) NEW_LINE
def countSubsequences ( arr ) : NEW_LINE
odd = 0 NEW_LINE
for x in arr : NEW_LINE
if ( x & 1 ) : NEW_LINE INDENT odd = odd + 1 NEW_LINE DEDENT
return ( 1 << odd ) - 1 NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 1 , 3 , 3 ] NEW_LINE DEDENT
print ( countSubsequences ( arr ) ) NEW_LINE
def getPairsCount ( arr , n ) : NEW_LINE
count = 0 NEW_LINE
for i in range ( n ) : NEW_LINE
for j in range ( arr [ i ] - ( i % arr [ i ] ) , n , arr [ i ] ) : NEW_LINE
if ( i < j and abs ( arr [ i ] - arr [ j ] ) >= min ( arr [ i ] , arr [ j ] ) ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT
return count NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 1 , 2 , 2 , 3 ] NEW_LINE N = len ( arr ) NEW_LINE print ( getPairsCount ( arr , N ) ) NEW_LINE DEDENT
def check ( N ) : NEW_LINE INDENT twos = 0 NEW_LINE fives = 0 NEW_LINE DEDENT
while ( N % 2 == 0 ) : NEW_LINE INDENT N /= 2 NEW_LINE twos += 1 NEW_LINE DEDENT
while ( N % 5 == 0 ) : NEW_LINE INDENT N /= 5 NEW_LINE fives += 1 NEW_LINE DEDENT if ( N == 1 and twos <= fives ) : NEW_LINE INDENT print ( 2 * fives - twos ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( - 1 ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 50 NEW_LINE check ( N ) NEW_LINE DEDENT
def rangeSum ( arr , N , L , R ) : NEW_LINE
sum = 0 NEW_LINE
for i in range ( L - 1 , R , 1 ) : NEW_LINE INDENT sum += arr [ i % N ] NEW_LINE DEDENT
print ( sum ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 5 , 2 , 6 , 9 ] NEW_LINE L = 10 NEW_LINE R = 13 NEW_LINE N = len ( arr ) NEW_LINE rangeSum ( arr , N , L , R ) NEW_LINE DEDENT
def rangeSum ( arr , N , L , R ) : NEW_LINE
prefix = [ 0 for i in range ( N + 1 ) ] NEW_LINE prefix [ 0 ] = 0 NEW_LINE
for i in range ( 1 , N + 1 , 1 ) : NEW_LINE INDENT prefix [ i ] = prefix [ i - 1 ] + arr [ i - 1 ] NEW_LINE DEDENT
leftsum = ( ( L - 1 ) // N ) * prefix [ N ] + prefix [ ( L - 1 ) % N ] NEW_LINE
rightsum = ( R // N ) * prefix [ N ] + prefix [ R % N ] NEW_LINE
print ( rightsum - leftsum ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 5 , 2 , 6 , 9 ] NEW_LINE L = 10 NEW_LINE R = 13 NEW_LINE N = len ( arr ) NEW_LINE rangeSum ( arr , N , L , R ) NEW_LINE DEDENT
def ExpoFactorial ( N ) : NEW_LINE
res = 1 NEW_LINE mod = ( int ) ( 1000000007 ) NEW_LINE
for i in range ( 2 , N + 1 ) : NEW_LINE
res = pow ( i , res , mod ) NEW_LINE
return res NEW_LINE
N = 4 NEW_LINE
print ( ExpoFactorial ( N ) ) NEW_LINE
def maxSubArraySumRepeated ( arr , N , K ) : NEW_LINE
sum = 0 NEW_LINE
for i in range ( N ) : NEW_LINE INDENT sum += arr [ i ] NEW_LINE DEDENT curr = arr [ 0 ] NEW_LINE
ans = arr [ 0 ] NEW_LINE
if ( K == 1 ) : NEW_LINE
for i in range ( 1 , N , 1 ) : NEW_LINE INDENT curr = max ( arr [ i ] , curr + arr [ i ] ) NEW_LINE ans = max ( ans , curr ) NEW_LINE DEDENT
return ans NEW_LINE
V = [ ] NEW_LINE
for i in range ( 2 * N ) : NEW_LINE INDENT V . append ( arr [ i % N ] ) NEW_LINE DEDENT
maxSuf = V [ 0 ] NEW_LINE
maxPref = V [ 2 * N - 1 ] NEW_LINE curr = V [ 0 ] NEW_LINE for i in range ( 1 , 2 * N , 1 ) : NEW_LINE INDENT curr += V [ i ] NEW_LINE maxPref = max ( maxPref , curr ) NEW_LINE DEDENT curr = V [ 2 * N - 1 ] NEW_LINE i = 2 * N - 2 NEW_LINE while ( i >= 0 ) : NEW_LINE INDENT curr += V [ i ] NEW_LINE maxSuf = max ( maxSuf , curr ) NEW_LINE i -= 1 NEW_LINE DEDENT curr = V [ 0 ] NEW_LINE
for i in range ( 1 , 2 * N , 1 ) : NEW_LINE INDENT curr = max ( V [ i ] , curr + V [ i ] ) NEW_LINE ans = max ( ans , curr ) NEW_LINE DEDENT
if ( sum > 0 ) : NEW_LINE INDENT temp = sum * ( K - 2 ) NEW_LINE ans = max ( ans , max ( temp + maxPref , temp + maxSuf ) ) NEW_LINE DEDENT
return ans NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
arr = [ 10 , 20 , - 30 , - 1 , 40 ] NEW_LINE N = len ( arr ) NEW_LINE K = 10 NEW_LINE
print ( maxSubArraySumRepeated ( arr , N , K ) ) NEW_LINE
def countSubarray ( arr , n ) : NEW_LINE
count = 0 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT for j in range ( i , n , 1 ) : NEW_LINE DEDENT
mxSubarray = 0 NEW_LINE
mxOther = 0 NEW_LINE
for k in range ( i , j + 1 , 1 ) : NEW_LINE INDENT mxSubarray = max ( mxSubarray , arr [ k ] ) NEW_LINE DEDENT
for k in range ( 0 , i , 1 ) : NEW_LINE INDENT mxOther = max ( mxOther , arr [ k ] ) NEW_LINE DEDENT for k in range ( j + 1 , n , 1 ) : NEW_LINE INDENT mxOther = max ( mxOther , arr [ k ] ) NEW_LINE DEDENT
if ( mxSubarray > ( 2 * mxOther ) ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT
print ( count ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 1 , 6 , 10 , 9 , 7 , 3 ] NEW_LINE N = len ( arr ) NEW_LINE countSubarray ( arr , N ) NEW_LINE DEDENT
def countSubarray ( arr , n ) : NEW_LINE INDENT count = 0 NEW_LINE L = 0 NEW_LINE R = 0 NEW_LINE DEDENT
mx = max ( arr ) NEW_LINE
for i in range ( n ) : NEW_LINE
if ( arr [ i ] * 2 > mx ) : NEW_LINE
L = i NEW_LINE break NEW_LINE i = n - 1 NEW_LINE while ( i >= 0 ) : NEW_LINE
if ( arr [ i ] * 2 > mx ) : NEW_LINE
R = i NEW_LINE break NEW_LINE i -= 1 NEW_LINE
print ( ( L + 1 ) * ( n - R ) ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 1 , 6 , 10 , 9 , 7 , 3 ] NEW_LINE N = len ( arr ) NEW_LINE countSubarray ( arr , N ) NEW_LINE DEDENT
from math import sqrt NEW_LINE
def isPrime ( X ) : NEW_LINE INDENT for i in range ( 2 , int ( sqrt ( X ) ) + 1 , 1 ) : NEW_LINE INDENT if ( X % i == 0 ) : NEW_LINE DEDENT DEDENT
return False NEW_LINE return True NEW_LINE
def printPrimes ( A , N ) : NEW_LINE
for i in range ( N ) : NEW_LINE
j = A [ i ] - 1 NEW_LINE while ( 1 ) : NEW_LINE
if ( isPrime ( j ) ) : NEW_LINE INDENT print ( j , end = " ▁ " ) NEW_LINE break NEW_LINE DEDENT j -= 1 NEW_LINE
j = A [ i ] + 1 NEW_LINE while ( 1 ) : NEW_LINE
if ( isPrime ( j ) ) : NEW_LINE INDENT print ( j , end = " ▁ " ) NEW_LINE break NEW_LINE DEDENT j += 1 NEW_LINE print ( " " , ▁ end ▁ = ▁ " " ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
A = [ 17 , 28 ] NEW_LINE N = len ( A ) NEW_LINE
printPrimes ( A , N ) NEW_LINE
def KthSmallest ( A , B , N , K ) : NEW_LINE INDENT M = 0 NEW_LINE DEDENT
for i in range ( N ) : NEW_LINE INDENT M = max ( A [ i ] , M ) NEW_LINE DEDENT
freq = [ 0 ] * ( M + 1 ) NEW_LINE
for i in range ( N ) : NEW_LINE INDENT freq [ A [ i ] ] += B [ i ] NEW_LINE DEDENT
sum = 0 NEW_LINE
for i in range ( M + 1 ) : NEW_LINE
sum += freq [ i ] NEW_LINE
if ( sum >= K ) : NEW_LINE
return i NEW_LINE
return - 1 NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
A = [ 3 , 4 , 5 ] NEW_LINE B = [ 2 , 1 , 3 ] NEW_LINE N = len ( A ) NEW_LINE K = 4 NEW_LINE
print ( KthSmallest ( A , B , N , K ) ) NEW_LINE
def findbitwiseOR ( a , n ) : NEW_LINE
res = 0 NEW_LINE
for i in range ( n ) : NEW_LINE
curr_sub_array = a [ i ] NEW_LINE
res = res | curr_sub_array NEW_LINE for j in range ( i , n ) : NEW_LINE
curr_sub_array = curr_sub_array & a [ j ] NEW_LINE res = res | curr_sub_array NEW_LINE
print ( res ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT A = [ 1 , 2 , 3 ] NEW_LINE N = len ( A ) NEW_LINE findbitwiseOR ( A , N ) NEW_LINE DEDENT
def findbitwiseOR ( a , n ) : NEW_LINE
res = 0 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT res = res | a [ i ] NEW_LINE DEDENT
print ( res ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT A = [ 1 , 2 , 3 ] NEW_LINE N = len ( A ) NEW_LINE findbitwiseOR ( A , N ) NEW_LINE DEDENT
def check ( n ) : NEW_LINE
sumOfDigit = 0 NEW_LINE prodOfDigit = 1 NEW_LINE while n > 0 : NEW_LINE
rem = n % 10 NEW_LINE
sumOfDigit += rem NEW_LINE
prodOfDigit *= rem NEW_LINE
n = n // 10 NEW_LINE
if sumOfDigit > prodOfDigit : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
N = 1234 NEW_LINE check ( N ) NEW_LINE
def evenOddBitwiseXOR ( N ) : NEW_LINE INDENT print ( " Even : ▁ " , 0 , end = " ▁ " ) NEW_LINE DEDENT
for i in range ( 4 , N + 1 , 4 ) : NEW_LINE INDENT print ( i , end = " ▁ " ) NEW_LINE DEDENT print ( ) NEW_LINE print ( " Odd : ▁ " , 1 , end = " ▁ " ) NEW_LINE
for i in range ( 4 , N + 1 , 4 ) : NEW_LINE INDENT print ( i - 1 , end = " ▁ " ) NEW_LINE DEDENT if ( N % 4 == 2 ) : NEW_LINE INDENT print ( N + 1 ) NEW_LINE DEDENT elif ( N % 4 == 3 ) : NEW_LINE INDENT print ( N ) NEW_LINE DEDENT
N = 6 NEW_LINE evenOddBitwiseXOR ( N ) NEW_LINE
def findPermutation ( arr ) : NEW_LINE INDENT N = len ( arr ) NEW_LINE i = N - 2 NEW_LINE DEDENT
while ( i >= 0 and arr [ i ] <= arr [ i + 1 ] ) : NEW_LINE INDENT i -= 1 NEW_LINE DEDENT
if ( i == - 1 ) : NEW_LINE INDENT print ( " - 1" ) NEW_LINE return NEW_LINE DEDENT j = N - 1 NEW_LINE
while ( j > i and arr [ j ] >= arr [ i ] ) : NEW_LINE INDENT j -= 1 NEW_LINE DEDENT
while ( j > i and arr [ j ] == arr [ j - 1 ] ) : NEW_LINE
j -= 1 NEW_LINE
temp = arr [ i ] ; NEW_LINE arr [ i ] = arr [ j ] ; NEW_LINE arr [ j ] = temp ; NEW_LINE
for it in arr : NEW_LINE INDENT print ( it , end = " ▁ " ) NEW_LINE DEDENT
arr = [ 1 , 2 , 5 , 3 , 4 , 6 ] NEW_LINE findPermutation ( arr ) NEW_LINE
def sieveOfEratosthenes ( N , s ) : NEW_LINE
prime = [ False ] * ( N + 1 ) NEW_LINE
for i in range ( 2 , N + 1 , 2 ) : NEW_LINE INDENT s [ i ] = 2 NEW_LINE DEDENT
for i in range ( 3 , N , 2 ) : NEW_LINE
if ( prime [ i ] == False ) : NEW_LINE INDENT s [ i ] = i NEW_LINE DEDENT
for j in range ( i , N , 2 ) : NEW_LINE INDENT if j * i > N : NEW_LINE INDENT break NEW_LINE DEDENT DEDENT
if ( not prime [ i * j ] ) : NEW_LINE INDENT prime [ i * j ] = True NEW_LINE s [ i * j ] = i NEW_LINE DEDENT
def findDifference ( N ) : NEW_LINE
s = [ 0 ] * ( N + 1 ) NEW_LINE
sieveOfEratosthenes ( N , s ) NEW_LINE
total , odd , even = 1 , 1 , 0 NEW_LINE
curr = s [ N ] NEW_LINE
cnt = 1 NEW_LINE
while ( N > 1 ) : NEW_LINE INDENT N //= s [ N ] NEW_LINE DEDENT
if ( curr == s [ N ] ) : NEW_LINE INDENT cnt += 1 NEW_LINE continue NEW_LINE DEDENT
if ( curr == 2 ) : NEW_LINE INDENT total = total * ( cnt + 1 ) NEW_LINE DEDENT
else : NEW_LINE INDENT total = total * ( cnt + 1 ) NEW_LINE odd = odd * ( cnt + 1 ) NEW_LINE DEDENT
curr = s [ N ] NEW_LINE cnt = 1 NEW_LINE
even = total - odd NEW_LINE
print ( abs ( even - odd ) ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 12 NEW_LINE findDifference ( N ) NEW_LINE DEDENT
def findMedian ( Mean , Mode ) : NEW_LINE
Median = ( 2 * Mean + Mode ) // 3 NEW_LINE
print ( Median ) NEW_LINE
Mode = 6 NEW_LINE Mean = 3 NEW_LINE findMedian ( Mean , Mode ) NEW_LINE
from math import sqrt NEW_LINE
def vectorMagnitude ( x , y , z ) : NEW_LINE
sum = x * x + y * y + z * z NEW_LINE
return sqrt ( sum ) NEW_LINE
x = 1 NEW_LINE y = 2 NEW_LINE z = 3 NEW_LINE print ( vectorMagnitude ( x , y , z ) ) NEW_LINE
import math NEW_LINE
def multiplyByMersenne ( N , M ) : NEW_LINE
x = int ( math . log2 ( M + 1 ) ) NEW_LINE
return ( ( N << x ) - N ) NEW_LINE
N = 4 NEW_LINE M = 15 NEW_LINE print ( multiplyByMersenne ( N , M ) ) NEW_LINE
from math import sqrt , log2 , pow NEW_LINE
def perfectSquare ( num ) : NEW_LINE
sr = int ( sqrt ( num ) ) NEW_LINE
a = sr * sr NEW_LINE b = ( sr + 1 ) * ( sr + 1 ) NEW_LINE
if ( ( num - a ) < ( b - num ) ) : NEW_LINE INDENT return a NEW_LINE DEDENT else : NEW_LINE INDENT return b NEW_LINE DEDENT
def powerOfTwo ( num ) : NEW_LINE
lg = int ( log2 ( num ) ) NEW_LINE
p = int ( pow ( 2 , lg ) ) NEW_LINE return p NEW_LINE
def uniqueElement ( arr , N ) : NEW_LINE INDENT ans = True NEW_LINE DEDENT
freq = { } NEW_LINE
for i in range ( N ) : NEW_LINE INDENT if ( arr [ i ] in freq ) : NEW_LINE INDENT freq [ arr [ i ] ] += 1 NEW_LINE DEDENT else : NEW_LINE INDENT freq [ arr [ i ] ] = 1 NEW_LINE DEDENT DEDENT
res = [ ] NEW_LINE for key , value in freq . items ( ) : NEW_LINE
if ( value == 1 ) : NEW_LINE INDENT ans = False NEW_LINE DEDENT
ps = perfectSquare ( key ) NEW_LINE
res . append ( powerOfTwo ( ps ) ) NEW_LINE res . sort ( reverse = False ) NEW_LINE for x in res : NEW_LINE print ( x , end = " ▁ " ) NEW_LINE
if ( ans ) : NEW_LINE INDENT print ( " - 1" ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 4 , 11 , 4 , 3 , 4 ] NEW_LINE N = len ( arr ) NEW_LINE uniqueElement ( arr , N ) NEW_LINE DEDENT
import sys NEW_LINE
def partitionArray ( a , n ) : NEW_LINE
INDENT Min = [ 0 ] * n NEW_LINE DEDENT
INDENT Mini = sys . maxsize NEW_LINE DEDENT
INDENT for i in range ( n - 1 , - 1 , - 1 ) : NEW_LINE DEDENT
Mini = min ( Mini , a [ i ] ) NEW_LINE
Min [ i ] = Mini NEW_LINE
INDENT Maxi = - sys . maxsize - 1 NEW_LINE DEDENT
INDENT ind = - 1 NEW_LINE for i in range ( n - 1 ) : NEW_LINE DEDENT
Maxi = max ( Maxi , a [ i ] ) NEW_LINE
if ( Maxi < Min [ i + 1 ] ) : NEW_LINE
ind = i NEW_LINE
break NEW_LINE
INDENT if ( ind != - 1 ) : NEW_LINE DEDENT
for i in range ( ind + 1 ) : NEW_LINE print ( a [ i ] , end = " ▁ " ) NEW_LINE print ( ) NEW_LINE
for i in range ( ind + 1 , n , 1 ) : NEW_LINE print ( a [ i ] , end = " ▁ " ) NEW_LINE
INDENT else : NEW_LINE INDENT print ( " Impossible " ) NEW_LINE DEDENT DEDENT
arr = [ 5 , 3 , 2 , 7 , 9 ] NEW_LINE N = 5 NEW_LINE partitionArray ( arr , N ) NEW_LINE
import math NEW_LINE
def countPrimeFactors ( n ) : NEW_LINE INDENT count = 0 NEW_LINE DEDENT
while ( n % 2 == 0 ) : NEW_LINE INDENT n = n // 2 NEW_LINE count += 1 NEW_LINE DEDENT
for i in range ( 3 , int ( math . sqrt ( n ) + 1 ) , 2 ) : NEW_LINE
while ( n % i == 0 ) : NEW_LINE INDENT n = n // i NEW_LINE count += 1 NEW_LINE DEDENT
if ( n > 2 ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT return ( count ) NEW_LINE
def findSum ( n ) : NEW_LINE
sum = 0 NEW_LINE i = 1 NEW_LINE num = 2 NEW_LINE while ( i <= n ) : NEW_LINE
if ( countPrimeFactors ( num ) == 2 ) : NEW_LINE INDENT sum += num NEW_LINE DEDENT
i += 1 NEW_LINE num += 1 NEW_LINE return sum NEW_LINE
def check ( n , k ) : NEW_LINE
s = findSum ( k - 1 ) NEW_LINE
if ( s >= n ) : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
else : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT
n = 100 NEW_LINE k = 6 NEW_LINE check ( n , k ) NEW_LINE
import math NEW_LINE
def gcd ( a , b ) : NEW_LINE
while ( b > 0 ) : NEW_LINE INDENT rem = a % b NEW_LINE a = b NEW_LINE b = rem NEW_LINE DEDENT
return a NEW_LINE
def countNumberOfWays ( n ) : NEW_LINE
if ( n == 1 ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
g = 0 NEW_LINE power = 0 NEW_LINE
while ( n % 2 == 0 ) : NEW_LINE INDENT power += 1 NEW_LINE n //= 2 NEW_LINE DEDENT g = gcd ( g , power ) NEW_LINE
for i in range ( 3 , int ( math . sqrt ( g ) ) + 1 , 2 ) : NEW_LINE INDENT power = 0 NEW_LINE DEDENT
while ( n % i == 0 ) : NEW_LINE INDENT power += 1 NEW_LINE n //= i NEW_LINE DEDENT g = gcd ( g , power ) NEW_LINE
if ( n > 2 ) : NEW_LINE INDENT g = gcd ( g , 1 ) NEW_LINE DEDENT
ways = 1 NEW_LINE
power = 0 NEW_LINE while ( g % 2 == 0 ) : NEW_LINE INDENT g //= 2 NEW_LINE power += 1 NEW_LINE DEDENT
ways *= ( power + 1 ) NEW_LINE
for i in range ( 3 , int ( math . sqrt ( g ) ) + 1 , 2 ) : NEW_LINE INDENT power = 0 NEW_LINE DEDENT
while ( g % i == 0 ) : NEW_LINE INDENT power += 1 NEW_LINE g /= i NEW_LINE DEDENT
ways *= ( power + 1 ) NEW_LINE
if ( g > 2 ) : NEW_LINE INDENT ways *= 2 NEW_LINE DEDENT
return ways NEW_LINE
N = 64 NEW_LINE print ( countNumberOfWays ( N ) ) NEW_LINE
from math import floor , ceil , log2 NEW_LINE
def powOfPositive ( n ) : NEW_LINE
pos = floor ( log2 ( n ) ) ; NEW_LINE return 2 ** pos ; NEW_LINE
def powOfNegative ( n ) : NEW_LINE
pos = ceil ( log2 ( n ) ) ; NEW_LINE return ( - 1 * pow ( 2 , pos ) ) ; NEW_LINE
def highestPowerOf2 ( n ) : NEW_LINE
if ( n > 0 ) : NEW_LINE INDENT print ( powOfPositive ( n ) ) ; NEW_LINE DEDENT else : NEW_LINE
n = - n ; NEW_LINE print ( powOfNegative ( n ) ) ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = - 24 ; NEW_LINE highestPowerOf2 ( n ) ; NEW_LINE DEDENT
def noOfCards ( n ) : NEW_LINE INDENT return n * ( 3 * n + 1 ) // 2 NEW_LINE DEDENT
n = 3 NEW_LINE print ( noOfCards ( n ) ) NEW_LINE
def smallestPoss ( s , n ) : NEW_LINE
ans = " " ; NEW_LINE
arr = [ 0 ] * 10 ; NEW_LINE
for i in range ( n ) : NEW_LINE INDENT arr [ ord ( s [ i ] ) - 48 ] += 1 ; NEW_LINE DEDENT
for i in range ( 10 ) : NEW_LINE INDENT for j in range ( arr [ i ] ) : NEW_LINE INDENT ans = ans + str ( i ) ; NEW_LINE DEDENT DEDENT
return ans ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 15 ; NEW_LINE K = "325343273113434" ; NEW_LINE print ( smallestPoss ( K , N ) ) ; NEW_LINE DEDENT
def Count_subarray ( arr , n ) : NEW_LINE INDENT subarray_sum , remaining_sum , count = 0 , 0 , 0 ; NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE
for j in range ( i , n ) : NEW_LINE
subarray_sum = 0 ; NEW_LINE remaining_sum = 0 ; NEW_LINE
for k in range ( i , j + 1 ) : NEW_LINE INDENT subarray_sum += arr [ k ] ; NEW_LINE DEDENT
for l in range ( i ) : NEW_LINE INDENT remaining_sum += arr [ l ] ; NEW_LINE DEDENT for l in range ( j + 1 , n ) : NEW_LINE INDENT remaining_sum += arr [ l ] ; NEW_LINE DEDENT
if ( subarray_sum > remaining_sum ) : NEW_LINE INDENT count += 1 ; NEW_LINE DEDENT return count ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 10 , 9 , 12 , 6 ] ; NEW_LINE n = len ( arr ) ; NEW_LINE print ( Count_subarray ( arr , n ) ) ; NEW_LINE DEDENT
def Count_subarray ( arr , n ) : NEW_LINE INDENT total_sum = 0 ; NEW_LINE count = 0 ; NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT total_sum += arr [ i ] ; NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE
subarray_sum = 0 ; NEW_LINE
for j in range ( i , n ) : NEW_LINE
subarray_sum += arr [ j ] ; NEW_LINE remaining_sum = total_sum - subarray_sum ; NEW_LINE
if ( subarray_sum > remaining_sum ) : NEW_LINE INDENT count += 1 ; NEW_LINE DEDENT return count ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 10 , 9 , 12 , 6 ] ; NEW_LINE n = len ( arr ) ; NEW_LINE print ( Count_subarray ( arr , n ) ) ; NEW_LINE DEDENT
def maxXOR ( arr , n ) : NEW_LINE
xorArr = 0 NEW_LINE for i in range ( n ) : NEW_LINE INDENT xorArr ^= arr [ i ] NEW_LINE DEDENT
ans = 0 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT ans = max ( ans , ( xorArr ^ arr [ i ] ) ) NEW_LINE DEDENT
return ans NEW_LINE
arr = [ 1 , 1 , 3 ] NEW_LINE n = len ( arr ) NEW_LINE print ( maxXOR ( arr , n ) ) NEW_LINE
def digitDividesK ( num , k ) : NEW_LINE INDENT while ( num ) : NEW_LINE DEDENT
d = num % 10 NEW_LINE
if ( d != 0 and k % d == 0 ) : NEW_LINE INDENT return True NEW_LINE DEDENT
num = num // 10 NEW_LINE
return False NEW_LINE
def findCount ( l , r , k ) : NEW_LINE
count = 0 NEW_LINE
for i in range ( l , r + 1 ) : NEW_LINE
if ( digitDividesK ( i , k ) ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT return count NEW_LINE
l = 20 NEW_LINE r = 35 NEW_LINE k = 45 NEW_LINE print ( findCount ( l , r , k ) ) NEW_LINE
def isFactorial ( n ) : NEW_LINE INDENT i = 1 ; NEW_LINE while ( True ) : NEW_LINE INDENT if ( n % i == 0 ) : NEW_LINE INDENT n //= i ; NEW_LINE DEDENT else : NEW_LINE INDENT break ; NEW_LINE DEDENT i += 1 ; NEW_LINE DEDENT if ( n == 1 ) : NEW_LINE INDENT return True ; NEW_LINE DEDENT else : NEW_LINE INDENT return False ; NEW_LINE DEDENT DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 24 ; NEW_LINE ans = isFactorial ( n ) ; NEW_LINE if ( ans == 1 ) : NEW_LINE INDENT print ( " Yes " ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) ; NEW_LINE DEDENT DEDENT
from math import gcd NEW_LINE
def lcm ( a , b ) : NEW_LINE INDENT GCD = gcd ( a , b ) ; NEW_LINE return ( a * b ) // GCD ; NEW_LINE DEDENT
def MinLCM ( a , n ) : NEW_LINE
Prefix = [ 0 ] * ( n + 2 ) ; NEW_LINE Suffix = [ 0 ] * ( n + 2 ) ; NEW_LINE
Prefix [ 1 ] = a [ 0 ] ; NEW_LINE for i in range ( 2 , n + 1 ) : NEW_LINE INDENT Prefix [ i ] = lcm ( Prefix [ i - 1 ] , a [ i - 1 ] ) ; NEW_LINE DEDENT
Suffix [ n ] = a [ n - 1 ] ; NEW_LINE
for i in range ( n - 1 , 0 , - 1 ) : NEW_LINE INDENT Suffix [ i ] = lcm ( Suffix [ i + 1 ] , a [ i - 1 ] ) ; NEW_LINE DEDENT
ans = min ( Suffix [ 2 ] , Prefix [ n - 1 ] ) ; NEW_LINE
for i in range ( 2 , n ) : NEW_LINE INDENT ans = min ( ans , lcm ( Prefix [ i - 1 ] , Suffix [ i + 1 ] ) ) ; NEW_LINE DEDENT
return ans ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = [ 5 , 15 , 9 , 36 ] ; NEW_LINE n = len ( a ) ; NEW_LINE print ( MinLCM ( a , n ) ) ; NEW_LINE DEDENT
def count ( n ) : NEW_LINE INDENT return n * ( 3 * n - 1 ) // 2 ; NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 3 ; NEW_LINE print ( count ( n ) ) ; NEW_LINE DEDENT
def findMinValue ( arr , n ) : NEW_LINE
sum = 0 NEW_LINE for i in range ( n ) : NEW_LINE INDENT sum += arr [ i ] NEW_LINE DEDENT
return ( sum // n ) + 1 NEW_LINE
arr = [ 4 , 2 , 1 , 10 , 6 ] NEW_LINE n = len ( arr ) NEW_LINE print ( findMinValue ( arr , n ) ) NEW_LINE
MOD = 1000000007 NEW_LINE
def modFact ( n , m ) : NEW_LINE INDENT result = 1 NEW_LINE for i in range ( 1 , m + 1 ) : NEW_LINE INDENT result = ( result * i ) % MOD NEW_LINE DEDENT return result NEW_LINE DEDENT
n = 3 NEW_LINE m = 2 NEW_LINE print ( modFact ( n , m ) ) NEW_LINE
mod = 10 ** 9 + 7 NEW_LINE
def power ( p ) : NEW_LINE INDENT res = 1 NEW_LINE for i in range ( 1 , p + 1 ) : NEW_LINE INDENT res *= 2 NEW_LINE res %= mod NEW_LINE DEDENT return res % mod NEW_LINE DEDENT
def subset_square_sum ( A ) : NEW_LINE INDENT n = len ( A ) NEW_LINE ans = 0 NEW_LINE DEDENT
for i in A : NEW_LINE INDENT ans += i * i % mod NEW_LINE ans %= mod NEW_LINE DEDENT return ans * power ( n - 1 ) % mod NEW_LINE
A = [ 3 , 7 ] NEW_LINE print ( subset_square_sum ( A ) ) NEW_LINE
N = 100050 NEW_LINE lpf = [ 0 for i in range ( N ) ] NEW_LINE mobius = [ 0 for i in range ( N ) ] NEW_LINE
def least_prime_factor ( ) : NEW_LINE INDENT for i in range ( 2 , N ) : NEW_LINE DEDENT
if ( lpf [ i ] == 0 ) : NEW_LINE INDENT for j in range ( i , N , i ) : NEW_LINE DEDENT
if ( lpf [ j ] == 0 ) : NEW_LINE INDENT lpf [ j ] = i NEW_LINE DEDENT
def Mobius ( ) : NEW_LINE INDENT for i in range ( 1 , N ) : NEW_LINE DEDENT
if ( i == 1 ) : NEW_LINE INDENT mobius [ i ] = 1 NEW_LINE DEDENT else : NEW_LINE
if ( lpf [ ( i // lpf [ i ] ) ] == lpf [ i ] ) : NEW_LINE INDENT mobius [ i ] = 0 NEW_LINE DEDENT
else : NEW_LINE INDENT mobius [ i ] = - 1 * mobius [ i // lpf [ i ] ] NEW_LINE DEDENT
def gcd_pairs ( a , n ) : NEW_LINE
maxi = 0 NEW_LINE
fre = [ 0 for i in range ( N ) ] NEW_LINE
for i in range ( n ) : NEW_LINE INDENT fre [ a [ i ] ] += 1 NEW_LINE maxi = max ( a [ i ] , maxi ) NEW_LINE DEDENT least_prime_factor ( ) NEW_LINE Mobius ( ) NEW_LINE
ans = 0 NEW_LINE
for i in range ( 1 , maxi + 1 ) : NEW_LINE INDENT if ( mobius [ i ] == 0 ) : NEW_LINE INDENT continue NEW_LINE DEDENT temp = 0 NEW_LINE for j in range ( i , maxi + 1 , i ) : NEW_LINE INDENT temp += fre [ j ] NEW_LINE DEDENT ans += temp * ( temp - 1 ) // 2 * mobius [ i ] NEW_LINE DEDENT
return ans NEW_LINE
a = [ 1 , 2 , 3 , 4 , 5 , 6 ] NEW_LINE n = len ( a ) NEW_LINE
print ( gcd_pairs ( a , n ) ) NEW_LINE
from math import log NEW_LINE
def compareVal ( x , y ) : NEW_LINE
a = y * log ( x ) ; NEW_LINE b = x * log ( y ) ; NEW_LINE
if ( a > b ) : NEW_LINE INDENT print ( x , " ^ " , y , " > " , y , " ^ " , x ) ; NEW_LINE DEDENT elif ( a < b ) : NEW_LINE INDENT print ( x , " ^ " , y , " < " , y , " ^ " , x ) ; NEW_LINE DEDENT elif ( a == b ) : NEW_LINE INDENT print ( x , " ^ " , y , " = " , y , " ^ " , x ) ; NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT x = 4 ; y = 5 ; NEW_LINE compareVal ( x , y ) ; NEW_LINE DEDENT
def ZigZag ( n ) : NEW_LINE
fact = [ 0 for i in range ( n + 1 ) ] NEW_LINE zig = [ 0 for i in range ( n + 1 ) ] NEW_LINE
fact [ 0 ] = 1 NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE INDENT fact [ i ] = fact [ i - 1 ] * i NEW_LINE DEDENT
zig [ 0 ] = 1 NEW_LINE zig [ 1 ] = 1 NEW_LINE print ( " zig ▁ zag ▁ numbers : ▁ " , end = " ▁ " ) NEW_LINE
print ( zig [ 0 ] , zig [ 1 ] , end = " ▁ " ) NEW_LINE
for i in range ( 2 , n ) : NEW_LINE INDENT sum = 0 NEW_LINE for k in range ( 0 , i ) : NEW_LINE DEDENT
sum += ( ( fact [ i - 1 ] // ( fact [ i - 1 - k ] * fact [ k ] ) ) * zig [ k ] * zig [ i - 1 - k ] ) NEW_LINE
zig [ i ] = sum // 2 NEW_LINE
print ( sum // 2 , end = " ▁ " ) NEW_LINE
n = 10 NEW_LINE
ZigZag ( n ) NEW_LINE
def find_count ( ele ) : NEW_LINE
count = 0 NEW_LINE for i in range ( len ( ele ) ) : NEW_LINE
p = [ ] NEW_LINE
c = 0 NEW_LINE j = len ( ele ) - 1 NEW_LINE
while j >= ( len ( ele ) - 1 - i ) and j >= 0 : NEW_LINE INDENT p . append ( ele [ j ] ) NEW_LINE j -= 1 NEW_LINE DEDENT j = len ( ele ) - 1 NEW_LINE k = 0 NEW_LINE
while j >= 0 : NEW_LINE
if ele [ j ] != p [ k ] : NEW_LINE INDENT break NEW_LINE DEDENT j -= 1 NEW_LINE k += 1 NEW_LINE
if k == len ( p ) : NEW_LINE INDENT c += 1 NEW_LINE k = 0 NEW_LINE DEDENT count = max ( count , c ) NEW_LINE
return count NEW_LINE
def solve ( n ) : NEW_LINE
count = 1 NEW_LINE
ele = [ ] NEW_LINE
for i in range ( n ) : NEW_LINE INDENT print ( count , end = " ▁ " ) NEW_LINE DEDENT
ele . append ( count ) NEW_LINE
count = find_count ( ele ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 10 NEW_LINE solve ( n ) NEW_LINE DEDENT
store = dict ( ) NEW_LINE
def Wedderburn ( n ) : NEW_LINE
if ( n <= 2 ) : NEW_LINE INDENT return store [ n ] NEW_LINE DEDENT
elif ( n % 2 == 0 ) : NEW_LINE
x = n // 2 NEW_LINE ans = 0 NEW_LINE
for i in range ( 1 , x ) : NEW_LINE INDENT ans += store [ i ] * store [ n - i ] NEW_LINE DEDENT
ans += ( store [ x ] * ( store [ x ] + 1 ) ) // 2 NEW_LINE
store [ n ] = ans NEW_LINE
return ans NEW_LINE else : NEW_LINE
x = ( n + 1 ) // 2 NEW_LINE ans = 0 NEW_LINE
for i in range ( 1 , x ) : NEW_LINE INDENT ans += store [ i ] * store [ n - i ] NEW_LINE DEDENT
store [ n ] = ans NEW_LINE
return ans NEW_LINE
def Wedderburn_Etherington ( n ) : NEW_LINE
store [ 0 ] = 0 NEW_LINE store [ 1 ] = 1 NEW_LINE store [ 2 ] = 1 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT print ( Wedderburn ( i ) , end = " " ) NEW_LINE if ( i != n - 1 ) : NEW_LINE INDENT print ( end = " , ▁ " ) NEW_LINE DEDENT DEDENT
n = 10 NEW_LINE
Wedderburn_Etherington ( n ) NEW_LINE
def Max_sum ( a , n ) : NEW_LINE
pos = 0 NEW_LINE neg = 0 NEW_LINE for i in range ( n ) : NEW_LINE
if ( a [ i ] > 0 ) : NEW_LINE INDENT pos = 1 NEW_LINE DEDENT
elif ( a [ i ] < 0 ) : NEW_LINE INDENT neg = 1 NEW_LINE DEDENT
if ( pos == 1 and neg == 1 ) : NEW_LINE INDENT break NEW_LINE DEDENT
sum = 0 NEW_LINE if ( pos == 1 and neg == 1 ) : NEW_LINE INDENT for i in range ( n ) : NEW_LINE INDENT sum += abs ( a [ i ] ) NEW_LINE DEDENT DEDENT elif ( pos == 1 ) : NEW_LINE
mini = a [ 0 ] NEW_LINE sum = a [ 0 ] NEW_LINE for i in range ( 1 , n , 1 ) : NEW_LINE INDENT mini = min ( mini , a [ i ] ) NEW_LINE sum += a [ i ] NEW_LINE DEDENT
sum -= 2 * mini NEW_LINE elif ( neg == 1 ) : NEW_LINE
for i in range ( n ) : NEW_LINE INDENT a [ i ] = abs ( a [ i ] ) NEW_LINE DEDENT
mini = a [ 0 ] NEW_LINE sum = a [ 0 ] NEW_LINE for i in range ( 1 , n ) : NEW_LINE INDENT mini = min ( mini , a [ i ] ) NEW_LINE sum += a [ i ] NEW_LINE DEDENT
sum -= 2 * mini NEW_LINE
return sum NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT a = [ 1 , 3 , 5 , - 2 , - 6 ] NEW_LINE n = len ( a ) NEW_LINE DEDENT
print ( Max_sum ( a , n ) ) NEW_LINE
def decimalToBinary ( n ) : NEW_LINE
if ( n == 0 ) : NEW_LINE INDENT print ( "0" , end = " " ) ; NEW_LINE return ; NEW_LINE DEDENT
decimalToBinary ( n // 2 ) ; NEW_LINE print ( n % 2 , end = " " ) ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 13 ; NEW_LINE decimalToBinary ( n ) ; NEW_LINE DEDENT
def MinimumValue ( x , y ) : NEW_LINE
if ( x > y ) : NEW_LINE INDENT x , y = y , x NEW_LINE DEDENT
a = 1 NEW_LINE b = x - 1 NEW_LINE c = y - b NEW_LINE print ( a , b , c ) NEW_LINE
x = 123 NEW_LINE y = 13 NEW_LINE
MinimumValue ( x , y ) NEW_LINE
def canConvert ( a , b ) : NEW_LINE INDENT while ( b > a ) : NEW_LINE DEDENT
if ( b % 10 == 1 ) : NEW_LINE INDENT b //= 10 ; NEW_LINE continue ; NEW_LINE DEDENT
if ( b % 2 == 0 ) : NEW_LINE INDENT b /= 2 ; NEW_LINE continue ; NEW_LINE DEDENT
return false ; NEW_LINE
if ( b == a ) : NEW_LINE INDENT return True ; NEW_LINE DEDENT return False ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT A = 2 ; B = 82 ; NEW_LINE if ( canConvert ( A , B ) ) : NEW_LINE INDENT print ( " Yes " ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) ; NEW_LINE DEDENT DEDENT
def count ( N ) : NEW_LINE INDENT a = 0 ; NEW_LINE a = ( N * ( N + 1 ) ) / 2 ; NEW_LINE return int ( a ) ; NEW_LINE DEDENT
N = 4 ; NEW_LINE print ( count ( N ) ) ; NEW_LINE
def numberOfDays ( a , b , n ) : NEW_LINE INDENT Days = b * ( n + a ) // ( a + b ) NEW_LINE return Days NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = 10 NEW_LINE b = 20 NEW_LINE n = 5 NEW_LINE print ( numberOfDays ( a , b , n ) ) NEW_LINE DEDENT
def getAverage ( x , y ) : NEW_LINE
avg = ( x & y ) + ( ( x ^ y ) >> 1 ) ; NEW_LINE return avg NEW_LINE
x = 10 NEW_LINE y = 9 NEW_LINE print ( getAverage ( x , y ) ) NEW_LINE
def smallestIndex ( a , n ) : NEW_LINE
right1 = 0 NEW_LINE right0 = 0 NEW_LINE
for i in range ( n ) : NEW_LINE
if ( a [ i ] == 1 ) : NEW_LINE INDENT right1 = i NEW_LINE DEDENT
else : NEW_LINE INDENT right0 = i NEW_LINE DEDENT
return min ( right1 , right0 ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT a = [ 1 , 1 , 1 , 0 , 0 , 1 , 0 , 1 , 1 ] NEW_LINE n = len ( a ) NEW_LINE print ( smallestIndex ( a , n ) ) NEW_LINE DEDENT
def countSquares ( r , c , m ) : NEW_LINE
squares = 0 NEW_LINE
for i in range ( 1 , 9 ) : NEW_LINE INDENT for j in range ( 1 , 9 ) : NEW_LINE DEDENT
if ( max ( abs ( i - r ) , abs ( j - c ) ) <= m ) : NEW_LINE INDENT squares = squares + 1 NEW_LINE DEDENT
return squares NEW_LINE
r = 4 NEW_LINE c = 4 NEW_LINE m = 1 NEW_LINE print ( countSquares ( r , c , m ) ) ; NEW_LINE
def countQuadruples ( a , n ) : NEW_LINE
mpp = dict . fromkeys ( a , 0 ) ; NEW_LINE
for i in range ( n ) : NEW_LINE INDENT mpp [ a [ i ] ] += 1 ; NEW_LINE DEDENT count = 0 ; NEW_LINE
for j in range ( n ) : NEW_LINE INDENT for k in range ( n ) : NEW_LINE DEDENT
if ( j == k ) : NEW_LINE INDENT continue ; NEW_LINE DEDENT
mpp [ a [ j ] ] -= 1 ; NEW_LINE mpp [ a [ k ] ] -= 1 ; NEW_LINE
first = a [ j ] - ( a [ k ] - a [ j ] ) ; NEW_LINE if first not in mpp : NEW_LINE INDENT mpp [ first ] = 0 ; NEW_LINE DEDENT
fourth = ( a [ k ] * a [ k ] ) // a [ j ] ; NEW_LINE if fourth not in mpp : NEW_LINE INDENT mpp [ fourth ] = 0 ; NEW_LINE DEDENT
if ( ( a [ k ] * a [ k ] ) % a [ j ] == 0 ) : NEW_LINE
if ( a [ j ] != a [ k ] ) : NEW_LINE INDENT count += mpp [ first ] * mpp [ fourth ] ; NEW_LINE DEDENT
else : NEW_LINE INDENT count += ( mpp [ first ] * ( mpp [ fourth ] - 1 ) ) ; NEW_LINE DEDENT
mpp [ a [ j ] ] += 1 ; NEW_LINE mpp [ a [ k ] ] += 1 ; NEW_LINE return count ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = [ 2 , 6 , 4 , 9 , 2 ] ; NEW_LINE n = len ( a ) ; NEW_LINE print ( countQuadruples ( a , n ) ) ; NEW_LINE DEDENT
def countNumbers ( L , R , K ) : NEW_LINE INDENT if ( K == 9 ) : NEW_LINE INDENT K = 0 NEW_LINE DEDENT DEDENT
totalnumbers = R - L + 1 NEW_LINE
factor9 = totalnumbers // 9 NEW_LINE
rem = totalnumbers % 9 NEW_LINE
ans = factor9 NEW_LINE
for i in range ( R , R - rem , - 1 ) : NEW_LINE INDENT rem1 = i % 9 NEW_LINE if ( rem1 == K ) : NEW_LINE INDENT ans += 1 NEW_LINE DEDENT DEDENT return ans NEW_LINE
L = 10 NEW_LINE R = 22 NEW_LINE K = 3 NEW_LINE print ( countNumbers ( L , R , K ) ) NEW_LINE
def EvenSum ( A , index , value ) : NEW_LINE
A [ index ] = A [ index ] + value NEW_LINE
sum = 0 NEW_LINE for i in A : NEW_LINE
if ( i % 2 == 0 ) : NEW_LINE INDENT sum = sum + i NEW_LINE DEDENT return sum NEW_LINE
def BalanceArray ( A , Q ) : NEW_LINE
ANS = [ ] NEW_LINE i , sum = 0 , 0 NEW_LINE for i in range ( len ( Q ) ) : NEW_LINE INDENT index = Q [ i ] [ 0 ] NEW_LINE value = Q [ i ] [ 1 ] NEW_LINE DEDENT
sum = EvenSum ( A , index , value ) NEW_LINE
ANS . append ( sum ) NEW_LINE
for i in ANS : NEW_LINE INDENT print ( i , end = " ▁ " ) NEW_LINE DEDENT
A = [ 1 , 2 , 3 , 4 ] NEW_LINE Q = [ [ 0 , 1 ] , [ 1 , - 3 ] , [ 0 , - 4 ] , [ 3 , 2 ] ] NEW_LINE BalanceArray ( A , Q ) NEW_LINE
def BalanceArray ( A , Q ) : NEW_LINE INDENT ANS = [ ] NEW_LINE sum = 0 NEW_LINE for i in range ( len ( A ) ) : NEW_LINE DEDENT
if ( A [ i ] % 2 == 0 ) : NEW_LINE INDENT sum += A [ i ] ; NEW_LINE DEDENT for i in range ( len ( Q ) ) : NEW_LINE index = Q [ i ] [ 0 ] ; NEW_LINE value = Q [ i ] [ 1 ] ; NEW_LINE
if ( A [ index ] % 2 == 0 ) : NEW_LINE INDENT sum -= A [ index ] ; NEW_LINE DEDENT A [ index ] += value ; NEW_LINE
if ( A [ index ] % 2 == 0 ) : NEW_LINE INDENT sum += A [ index ] ; NEW_LINE DEDENT
ANS . append ( sum ) ; NEW_LINE
for i in range ( len ( ANS ) ) : NEW_LINE INDENT print ( ANS [ i ] , end = " ▁ " ) ; NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT A = [ 1 , 2 , 3 , 4 ] ; NEW_LINE Q = [ [ 0 , 1 ] , [ 1 , - 3 ] , [ 0 , - 4 ] , [ 3 , 2 ] ] ; NEW_LINE BalanceArray ( A , Q ) ; NEW_LINE DEDENT
import math as mt NEW_LINE
def Cycles ( N ) : NEW_LINE INDENT fact = 1 NEW_LINE result = N - 1 NEW_LINE DEDENT
i = result NEW_LINE while ( i > 0 ) : NEW_LINE INDENT fact = fact * i NEW_LINE i -= 1 NEW_LINE DEDENT return fact // 2 NEW_LINE
N = 5 NEW_LINE Number = Cycles ( N ) NEW_LINE print ( " Hamiltonian ▁ cycles ▁ = ▁ " , Number ) NEW_LINE
def digitWell ( n , m , k ) : NEW_LINE INDENT cnt = 0 NEW_LINE while ( n > 0 ) : NEW_LINE INDENT if ( n % 10 == m ) : NEW_LINE INDENT cnt = cnt + 1 ; NEW_LINE DEDENT n = ( int ) ( n / 10 ) ; NEW_LINE DEDENT return cnt == k ; NEW_LINE DEDENT
def findInt ( n , m , k ) : NEW_LINE INDENT i = n + 1 ; NEW_LINE while ( True ) : NEW_LINE INDENT if ( digitWell ( i , m , k ) ) : NEW_LINE INDENT return i ; NEW_LINE DEDENT i = i + 1 ; NEW_LINE DEDENT DEDENT
n = 111 ; m = 2 ; k = 2 ; NEW_LINE print ( findInt ( n , m , k ) ) ; NEW_LINE
def countOdd ( arr , n ) : NEW_LINE
odd = 0 ; NEW_LINE for i in range ( 0 , n ) : NEW_LINE
if ( arr [ i ] % 2 == 1 ) : NEW_LINE INDENT odd = odd + 1 ; NEW_LINE DEDENT return odd ; NEW_LINE
def countValidPairs ( arr , n ) : NEW_LINE INDENT odd = countOdd ( arr , n ) ; NEW_LINE return ( odd * ( odd - 1 ) ) / 2 ; NEW_LINE DEDENT
arr = [ 1 , 2 , 3 , 4 , 5 ] ; NEW_LINE n = len ( arr ) ; NEW_LINE print ( int ( countValidPairs ( arr , n ) ) ) ; NEW_LINE
def gcd ( a , b ) : NEW_LINE INDENT if ( b == 0 ) : NEW_LINE INDENT return a NEW_LINE DEDENT else : NEW_LINE INDENT return gcd ( b , a % b ) NEW_LINE DEDENT DEDENT
def lcmOfArray ( arr , n ) : NEW_LINE INDENT if ( n < 1 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT lcm = arr [ 0 ] NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT lcm = ( lcm * arr [ i ] ) // gcd ( lcm , arr [ i ] ) ; NEW_LINE DEDENT
return lcm NEW_LINE
def minPerfectCube ( arr , n ) : NEW_LINE
lcm = lcmOfArray ( arr , n ) NEW_LINE minPerfectCube = lcm NEW_LINE cnt = 0 NEW_LINE while ( lcm > 1 and lcm % 2 == 0 ) : NEW_LINE INDENT cnt += 1 NEW_LINE lcm //= 2 NEW_LINE DEDENT
if ( cnt % 3 == 2 ) : NEW_LINE INDENT minPerfectCube *= 2 NEW_LINE DEDENT elif ( cnt % 3 == 1 ) : NEW_LINE INDENT minPerfectCube *= 4 NEW_LINE DEDENT i = 3 NEW_LINE
while ( lcm > 1 ) : NEW_LINE INDENT cnt = 0 NEW_LINE while ( lcm % i == 0 ) : NEW_LINE INDENT cnt += 1 NEW_LINE lcm //= i NEW_LINE DEDENT if ( cnt % 3 == 1 ) : NEW_LINE INDENT minPerfectCube *= i * i NEW_LINE DEDENT elif ( cnt % 3 == 2 ) : NEW_LINE INDENT minPerfectCube *= i NEW_LINE DEDENT i += 2 NEW_LINE DEDENT
return minPerfectCube NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 10 , 125 , 14 , 42 , 100 ] NEW_LINE n = len ( arr ) NEW_LINE print ( minPerfectCube ( arr , n ) ) NEW_LINE DEDENT
from math import sqrt NEW_LINE
def isPrime ( n ) : NEW_LINE
if ( n <= 1 ) : NEW_LINE INDENT return False NEW_LINE DEDENT if ( n <= 3 ) : NEW_LINE INDENT return True NEW_LINE DEDENT
if ( n % 2 == 0 or n % 3 == 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT k = int ( sqrt ( n ) ) + 1 NEW_LINE for i in range ( 5 , k , 6 ) : NEW_LINE INDENT if ( n % i == 0 or n % ( i + 2 ) == 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT return True NEW_LINE
def isStrongPrime ( n ) : NEW_LINE
if ( isPrime ( n ) == False or n == 2 ) : NEW_LINE INDENT return False NEW_LINE DEDENT
previous_prime = n - 1 NEW_LINE next_prime = n + 1 NEW_LINE
while ( isPrime ( next_prime ) == False ) : NEW_LINE INDENT next_prime += 1 NEW_LINE DEDENT
while ( isPrime ( previous_prime ) == False ) : NEW_LINE INDENT previous_prime -= 1 NEW_LINE DEDENT
mean = ( previous_prime + next_prime ) / 2 NEW_LINE
if ( n > mean ) : NEW_LINE INDENT return True NEW_LINE DEDENT else : NEW_LINE INDENT return False NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 11 NEW_LINE if ( isStrongPrime ( n ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT DEDENT
def countDigitsToBeRemoved ( N , K ) : NEW_LINE
s = str ( N ) ; NEW_LINE
res = 0 ; NEW_LINE
f_zero = 0 ; NEW_LINE for i in range ( len ( s ) - 1 , - 1 , - 1 ) : NEW_LINE INDENT if ( K == 0 ) : NEW_LINE INDENT return res ; NEW_LINE DEDENT if ( s [ i ] == '0' ) : NEW_LINE DEDENT
f_zero = 1 ; NEW_LINE K -= 1 ; NEW_LINE else : NEW_LINE res += 1 ; NEW_LINE
if ( K == 0 ) : NEW_LINE INDENT return res ; NEW_LINE DEDENT elif ( f_zero > 0 ) : NEW_LINE INDENT return len ( s ) - 1 ; NEW_LINE DEDENT return - 1 ; NEW_LINE
N = 10904025 ; NEW_LINE K = 2 ; NEW_LINE print ( countDigitsToBeRemoved ( N , K ) ) ; NEW_LINE N = 1000 ; NEW_LINE K = 5 ; NEW_LINE print ( countDigitsToBeRemoved ( N , K ) ) ; NEW_LINE N = 23985 ; NEW_LINE K = 2 ; NEW_LINE print ( countDigitsToBeRemoved ( N , K ) ) ; NEW_LINE
import math NEW_LINE
def getSum ( a , n ) : NEW_LINE
sum = 0 ; NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE
sum += ( i / math . pow ( a , i ) ) ; NEW_LINE return sum ; NEW_LINE
a = 3 ; n = 3 ; NEW_LINE
print ( getSum ( a , n ) ) ; NEW_LINE
from math import sqrt NEW_LINE
def largestPrimeFactor ( n ) : NEW_LINE
max = - 1 NEW_LINE
while n % 2 == 0 : NEW_LINE INDENT max = 2 ; NEW_LINE DEDENT
for i in range ( 3 , int ( sqrt ( n ) ) + 1 , 2 ) : NEW_LINE INDENT while n % i == 0 : NEW_LINE INDENT max = i ; NEW_LINE n = n / i ; NEW_LINE DEDENT DEDENT
if n > 2 : NEW_LINE INDENT max = n NEW_LINE DEDENT return max NEW_LINE
def checkUnusual ( n ) : NEW_LINE
factor = largestPrimeFactor ( n ) NEW_LINE
if factor > sqrt ( n ) : NEW_LINE INDENT return True NEW_LINE DEDENT else : NEW_LINE INDENT return False NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 14 NEW_LINE if checkUnusual ( n ) : NEW_LINE INDENT print ( " YES " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NO " ) NEW_LINE DEDENT DEDENT
def isHalfReducible ( arr , n , m ) : NEW_LINE INDENT frequencyHash = [ 0 ] * ( m + 1 ) ; NEW_LINE i = 0 ; NEW_LINE while ( i < n ) : NEW_LINE INDENT frequencyHash [ ( arr [ i ] % ( m + 1 ) ) ] += 1 ; NEW_LINE i += 1 ; NEW_LINE DEDENT i = 0 ; NEW_LINE while ( i <= m ) : NEW_LINE INDENT if ( frequencyHash [ i ] >= ( n / 2 ) ) : NEW_LINE INDENT break ; NEW_LINE DEDENT i += 1 ; NEW_LINE DEDENT if ( i <= m ) : NEW_LINE INDENT print ( " Yes " ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) ; NEW_LINE DEDENT DEDENT
arr = [ 8 , 16 , 32 , 3 , 12 ] ; NEW_LINE n = len ( arr ) ; NEW_LINE m = 7 ; NEW_LINE isHalfReducible ( arr , n , m ) ; NEW_LINE
arr = [ ] NEW_LINE
def generateDivisors ( n ) : NEW_LINE
for i in range ( 1 , int ( n ** ( 0.5 ) ) + 1 ) : NEW_LINE INDENT if n % i == 0 : NEW_LINE DEDENT
' NEW_LINE INDENT if n // i == i : NEW_LINE INDENT arr . append ( i ) NEW_LINE DEDENT DEDENT
else : NEW_LINE INDENT arr . append ( i ) NEW_LINE arr . append ( n // i ) NEW_LINE DEDENT
def harmonicMean ( n ) : NEW_LINE INDENT generateDivisors ( n ) NEW_LINE DEDENT
Sum = 0 NEW_LINE length = len ( arr ) NEW_LINE
for i in range ( 0 , length ) : NEW_LINE INDENT Sum = Sum + ( n / arr [ i ] ) NEW_LINE DEDENT Sum = Sum / n NEW_LINE
return length / Sum NEW_LINE
def isOreNumber ( n ) : NEW_LINE
mean = harmonicMean ( n ) NEW_LINE
if mean - int ( mean ) == 0 : NEW_LINE INDENT return True NEW_LINE DEDENT else : NEW_LINE INDENT return False NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 28 NEW_LINE if isOreNumber ( n ) == True : NEW_LINE INDENT print ( " YES " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NO " ) NEW_LINE DEDENT DEDENT
MAX = 10000 NEW_LINE s = set ( ) NEW_LINE
def SieveOfEratosthenes ( ) : NEW_LINE
prime = [ True ] * ( MAX ) NEW_LINE prime [ 0 ] , prime [ 1 ] = False , False NEW_LINE for p in range ( 2 , 100 ) : NEW_LINE
if prime [ p ] == True : NEW_LINE
for i in range ( p * 2 , MAX , p ) : NEW_LINE INDENT prime [ i ] = False NEW_LINE DEDENT
' NEW_LINE INDENT product = 1 NEW_LINE for p in range ( 2 , MAX ) : NEW_LINE INDENT if prime [ p ] == True : NEW_LINE DEDENT DEDENT
product = product * p NEW_LINE
s . add ( product + 1 ) NEW_LINE
def isEuclid ( n ) : NEW_LINE
if n in s : NEW_LINE INDENT return True NEW_LINE DEDENT else : NEW_LINE INDENT return False NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE
SieveOfEratosthenes ( ) NEW_LINE
n = 31 NEW_LINE
if isEuclid ( n ) == True : NEW_LINE INDENT print ( " YES " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NO " ) NEW_LINE DEDENT
n = 42 NEW_LINE
if isEuclid ( n ) == True : NEW_LINE INDENT print ( " YES " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NO " ) NEW_LINE DEDENT
def isPrime ( n ) : NEW_LINE
if ( n <= 1 ) : NEW_LINE INDENT return False NEW_LINE DEDENT if ( n <= 3 ) : NEW_LINE INDENT return True NEW_LINE DEDENT
if ( n % 2 == 0 or n % 3 == 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT i = 5 NEW_LINE while ( i * i <= n ) : NEW_LINE INDENT if ( n % i == 0 or n % ( i + 2 ) == 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT i = i + 6 NEW_LINE DEDENT return True NEW_LINE
def isPowerOfTwo ( n ) : NEW_LINE INDENT return ( n and ( not ( n & ( n - 1 ) ) ) ) NEW_LINE DEDENT
n = 43 NEW_LINE
if ( isPrime ( n ) and isPowerOfTwo ( n * 3 - 1 ) ) : NEW_LINE INDENT print ( " YES " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NO " ) NEW_LINE DEDENT
from math import pow , sqrt NEW_LINE
def area ( a ) : NEW_LINE
if ( a < 0 ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
area = pow ( ( a * sqrt ( 3 ) ) / ( sqrt ( 2 ) ) , 2 ) NEW_LINE return area NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT a = 5 NEW_LINE print ( " { 0 : . 3 } " . format ( area ( a ) ) ) NEW_LINE DEDENT
def nthTerm ( n ) : NEW_LINE INDENT return 3 * pow ( n , 2 ) - 4 * n + 2 NEW_LINE DEDENT
N = 4 NEW_LINE print ( nthTerm ( N ) ) NEW_LINE
def calculateSum ( n ) : NEW_LINE INDENT return ( n * ( n + 1 ) // 2 + pow ( ( n * ( n + 1 ) // 2 ) , 2 ) ) NEW_LINE DEDENT
n = 3 NEW_LINE
print ( " Sum ▁ = ▁ " , calculateSum ( n ) ) NEW_LINE
def arePermutations ( a , b , n , m ) : NEW_LINE INDENT sum1 , sum2 , mul1 , mul2 = 0 , 0 , 1 , 1 NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT sum1 += a [ i ] NEW_LINE mul1 *= a [ i ] NEW_LINE DEDENT
for i in range ( m ) : NEW_LINE INDENT sum2 += b [ i ] NEW_LINE mul2 *= b [ i ] NEW_LINE DEDENT
return ( ( sum1 == sum2 ) and ( mul1 == mul2 ) ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = [ 1 , 3 , 2 ] NEW_LINE b = [ 3 , 1 , 2 ] NEW_LINE n = len ( a ) NEW_LINE m = len ( b ) NEW_LINE if arePermutations ( a , b , n , m ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT DEDENT
def Race ( B , C ) : NEW_LINE INDENT result = 0 ; NEW_LINE DEDENT
result = ( ( C * 100 ) // B ) NEW_LINE return 100 - result NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT B = 10 NEW_LINE C = 28 NEW_LINE DEDENT
B = 100 - B ; NEW_LINE C = 100 - C ; NEW_LINE print ( str ( Race ( B , C ) ) + " ▁ meters " ) NEW_LINE
def Time ( arr , n , Emptypipe ) : NEW_LINE INDENT fill = 0 NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT fill += ( 1 / arr [ i ] ) NEW_LINE DEDENT fill = fill - ( 1 / float ( Emptypipe ) ) NEW_LINE return int ( 1 / fill ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 12 , 14 ] NEW_LINE Emptypipe = 30 NEW_LINE n = len ( arr ) NEW_LINE print ( ( Time ( arr , n , Emptypipe ) ) , " Hours " ) NEW_LINE DEDENT
def check ( n ) : NEW_LINE INDENT sum = 0 NEW_LINE DEDENT
while n != 0 : NEW_LINE INDENT sum += n % 10 NEW_LINE n = n // 10 NEW_LINE DEDENT
if sum % 7 == 0 : NEW_LINE INDENT return 1 NEW_LINE DEDENT else : NEW_LINE INDENT return 0 NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE
n = 25 NEW_LINE print ( ( " YES " ) if check ( n ) == 1 else print ( " NO " ) ) NEW_LINE
N = 1000005 NEW_LINE
def isPrime ( n ) : NEW_LINE
if n <= 1 : NEW_LINE INDENT return False NEW_LINE DEDENT if n <= 3 : NEW_LINE INDENT return True NEW_LINE DEDENT
if n % 2 == 0 or n % 3 == 0 : NEW_LINE INDENT return False NEW_LINE DEDENT i = 5 NEW_LINE while i * i <= n : NEW_LINE INDENT if ( n % i == 0 or n % ( i + 2 ) == 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT i = i + 6 NEW_LINE DEDENT return True NEW_LINE
def SumOfPrimeDivisors ( n ) : NEW_LINE INDENT sum = 0 NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE INDENT if n % i == 0 : NEW_LINE INDENT if isPrime ( i ) : NEW_LINE INDENT sum += i NEW_LINE DEDENT DEDENT DEDENT return sum NEW_LINE DEDENT
n = 60 NEW_LINE print ( " Sum ▁ of ▁ prime ▁ divisors ▁ of ▁ 60 ▁ is ▁ " + str ( SumOfPrimeDivisors ( n ) ) ) NEW_LINE
def Sum ( N ) : NEW_LINE INDENT SumOfPrimeDivisors = [ 0 ] * ( N + 1 ) NEW_LINE for i in range ( 2 , N + 1 ) : NEW_LINE DEDENT
if ( SumOfPrimeDivisors [ i ] == 0 ) : NEW_LINE
for j in range ( i , N + 1 , i ) : NEW_LINE INDENT SumOfPrimeDivisors [ j ] += i NEW_LINE DEDENT return SumOfPrimeDivisors [ N ] NEW_LINE
N = 60 NEW_LINE print ( " Sum ▁ of ▁ prime " , " divisors ▁ of ▁ 60 ▁ is " , Sum ( N ) ) ; NEW_LINE
def power ( x , y , p ) : NEW_LINE
x = x % p NEW_LINE while ( y > 0 ) : NEW_LINE
if ( y & 1 ) : NEW_LINE INDENT res = ( res * x ) % p NEW_LINE DEDENT
x = ( x * x ) % p NEW_LINE return res NEW_LINE
a = 3 NEW_LINE
b = "100000000000000000000000000" NEW_LINE remainderB = 0 NEW_LINE MOD = 1000000007 NEW_LINE
for i in range ( len ( b ) ) : NEW_LINE INDENT remainderB = ( ( remainderB * 10 + ord ( b [ i ] ) - 48 ) % ( MOD - 1 ) ) NEW_LINE DEDENT print ( power ( a , remainderB , MOD ) ) NEW_LINE
def find_Square_369 ( num ) : NEW_LINE
if ( num [ 0 ] == '3' ) : NEW_LINE INDENT a = '1' NEW_LINE b = '0' NEW_LINE c = '8' NEW_LINE d = '9' NEW_LINE DEDENT
elif ( num [ 0 ] == '6' ) : NEW_LINE INDENT a = '4' NEW_LINE b = '3' NEW_LINE c = '5' NEW_LINE d = '6' NEW_LINE DEDENT
else : NEW_LINE INDENT a = '9' NEW_LINE b = '8' NEW_LINE c = '0' NEW_LINE d = '1' NEW_LINE DEDENT
result = " " NEW_LINE
size = len ( num ) NEW_LINE
for i in range ( 1 , size ) : NEW_LINE INDENT result += a NEW_LINE DEDENT
result += b NEW_LINE
for i in range ( 1 , size ) : NEW_LINE INDENT result += c NEW_LINE DEDENT
result += d NEW_LINE
return result NEW_LINE
num_3 = "3333" NEW_LINE num_6 = "6666" NEW_LINE num_9 = "9999" NEW_LINE result = " " NEW_LINE
result = find_Square_369 ( num_3 ) NEW_LINE print ( " Square ▁ of ▁ " + num_3 + " ▁ is ▁ : ▁ " + result ) ; NEW_LINE
result = find_Square_369 ( num_6 ) NEW_LINE print ( " Square ▁ of ▁ " + num_6 + " ▁ is ▁ : ▁ " + result ) ; NEW_LINE
result = find_Square_369 ( num_9 ) NEW_LINE print ( " Square ▁ of ▁ " + num_9 + " ▁ is ▁ : ▁ " + result ) ; NEW_LINE
ans = 1 NEW_LINE mod = 1000000007 * 120 NEW_LINE for i in range ( 0 , 5 ) : NEW_LINE INDENT ans = ( ans * ( 55555 - i ) ) % mod NEW_LINE DEDENT ans = int ( ans / 120 ) NEW_LINE print ( " Answer ▁ using ▁ shortcut : ▁ " , ans ) NEW_LINE
def fact ( n ) : NEW_LINE INDENT if ( n == 0 or n == 1 ) : NEW_LINE INDENT return 1 ; NEW_LINE DEDENT ans = 1 ; NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE INDENT ans = ans * i ; NEW_LINE DEDENT return ans ; NEW_LINE DEDENT
def nCr ( n , r ) : NEW_LINE INDENT Nr = n ; Dr = 1 ; ans = 1 ; NEW_LINE for i in range ( 1 , r + 1 ) : NEW_LINE INDENT ans = int ( ( ans * Nr ) / ( Dr ) ) ; NEW_LINE Nr = Nr - 1 ; NEW_LINE Dr = Dr + 1 ; NEW_LINE DEDENT return ans ; NEW_LINE DEDENT
def solve ( n ) : NEW_LINE INDENT N = 2 * n - 2 ; NEW_LINE R = n - 1 ; NEW_LINE return ( nCr ( N , R ) * fact ( n - 1 ) ) ; NEW_LINE DEDENT
n = 6 ; NEW_LINE print ( solve ( n ) ) ; NEW_LINE
def pythagoreanTriplet ( n ) : NEW_LINE
for i in range ( 1 , int ( n / 3 ) + 1 ) : NEW_LINE
for j in range ( i + 1 , int ( n / 2 ) + 1 ) : NEW_LINE INDENT k = n - i - j NEW_LINE if ( i * i + j * j == k * k ) : NEW_LINE INDENT print ( i , " , ▁ " , j , " , ▁ " , k , sep = " " ) NEW_LINE return NEW_LINE DEDENT DEDENT print ( " No ▁ Triplet " ) NEW_LINE
n = 12 NEW_LINE pythagoreanTriplet ( n ) NEW_LINE
def factorial ( n ) : NEW_LINE INDENT f = 1 NEW_LINE for i in range ( 2 , n + 1 ) : NEW_LINE INDENT f *= i NEW_LINE DEDENT return f NEW_LINE DEDENT
def series ( A , X , n ) : NEW_LINE
nFact = factorial ( n ) NEW_LINE
for i in range ( 0 , n + 1 ) : NEW_LINE
niFact = factorial ( n - i ) NEW_LINE iFact = factorial ( i ) NEW_LINE
aPow = pow ( A , n - i ) NEW_LINE xPow = pow ( X , i ) NEW_LINE
print ( int ( ( nFact * aPow * xPow ) / ( niFact * iFact ) ) , end = " ▁ " ) NEW_LINE
A = 3 ; X = 4 ; n = 5 NEW_LINE series ( A , X , n ) NEW_LINE
def seiresSum ( n , a ) : NEW_LINE INDENT res = 0 NEW_LINE for i in range ( 0 , 2 * n ) : NEW_LINE INDENT if ( i % 2 == 0 ) : NEW_LINE INDENT res += a [ i ] * a [ i ] NEW_LINE DEDENT else : NEW_LINE INDENT res -= a [ i ] * a [ i ] NEW_LINE DEDENT DEDENT return res NEW_LINE DEDENT
n = 2 NEW_LINE a = [ 1 , 2 , 3 , 4 ] NEW_LINE print ( seiresSum ( n , a ) ) NEW_LINE
def power ( n , r ) : NEW_LINE
count = 0 ; i = r NEW_LINE while ( ( n / i ) >= 1 ) : NEW_LINE INDENT count += n / i NEW_LINE i = i * r NEW_LINE DEDENT return int ( count ) NEW_LINE
n = 6 ; r = 3 NEW_LINE print ( power ( n , r ) ) NEW_LINE
def avg_of_odd_num ( n ) : NEW_LINE
sm = 0 NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT sm = sm + ( 2 * i + 1 ) NEW_LINE DEDENT
return sm // n NEW_LINE
n = 20 NEW_LINE print ( avg_of_odd_num ( n ) ) NEW_LINE
def avg_of_odd_num ( n ) : NEW_LINE INDENT return n NEW_LINE DEDENT
n = 8 NEW_LINE print ( avg_of_odd_num ( n ) ) NEW_LINE
def fib ( f , N ) : NEW_LINE
f [ 1 ] = 1 NEW_LINE f [ 2 ] = 1 NEW_LINE for i in range ( 3 , N + 1 ) : NEW_LINE
f [ i ] = f [ i - 1 ] + f [ i - 2 ] NEW_LINE def fiboTriangle ( n ) : NEW_LINE
N = n * ( n + 1 ) // 2 NEW_LINE f = [ 0 ] * ( N + 1 ) NEW_LINE fib ( f , N ) NEW_LINE
fiboNum = 1 NEW_LINE
for i in range ( 1 , n + 1 ) : NEW_LINE
for j in range ( 1 , i + 1 ) : NEW_LINE INDENT print ( f [ fiboNum ] , " ▁ " , end = " " ) NEW_LINE fiboNum = fiboNum + 1 NEW_LINE DEDENT print ( ) NEW_LINE
n = 5 NEW_LINE fiboTriangle ( n ) NEW_LINE
def averageOdd ( n ) : NEW_LINE INDENT if ( n % 2 == 0 ) : NEW_LINE INDENT print ( " Invalid ▁ Input " ) NEW_LINE return - 1 NEW_LINE DEDENT sm = 0 NEW_LINE count = 0 NEW_LINE while ( n >= 1 ) : NEW_LINE DEDENT
count = count + 1 NEW_LINE
sm = sm + n NEW_LINE n = n - 2 NEW_LINE return sm // count NEW_LINE
n = 15 NEW_LINE print ( averageOdd ( n ) ) NEW_LINE
def averageOdd ( n ) : NEW_LINE INDENT if ( n % 2 == 0 ) : NEW_LINE INDENT print ( " Invalid ▁ Input " ) NEW_LINE return - 1 NEW_LINE DEDENT return ( n + 1 ) // 2 NEW_LINE DEDENT
n = 15 NEW_LINE print ( averageOdd ( n ) ) NEW_LINE
import math NEW_LINE
def lcm ( a , b ) : NEW_LINE INDENT return ( a * b ) // ( math . gcd ( a , b ) ) NEW_LINE DEDENT
def maxRational ( first , sec ) : NEW_LINE
k = lcm ( first [ 1 ] , sec [ 1 ] ) NEW_LINE
nume1 = first [ 0 ] NEW_LINE nume2 = sec [ 0 ] NEW_LINE nume1 *= k // ( first [ 1 ] ) NEW_LINE nume2 *= k // ( sec [ 1 ] ) NEW_LINE return first if ( nume2 < nume1 ) else sec NEW_LINE
first = [ 3 , 2 ] NEW_LINE sec = [ 3 , 4 ] NEW_LINE res = maxRational ( first , sec ) NEW_LINE print ( res [ 0 ] , " / " , res [ 1 ] , sep = " " ) NEW_LINE
def TrinomialValue ( n , k ) : NEW_LINE
if n == 0 and k == 0 : NEW_LINE INDENT return 1 NEW_LINE DEDENT
if k < - n or k > n : NEW_LINE INDENT return 0 NEW_LINE DEDENT
return ( TrinomialValue ( n - 1 , k - 1 ) + TrinomialValue ( n - 1 , k ) + TrinomialValue ( n - 1 , k + 1 ) ) NEW_LINE
def printTrinomial ( n ) : NEW_LINE
for i in range ( n ) : NEW_LINE
for j in range ( - i , 1 ) : NEW_LINE INDENT print ( TrinomialValue ( i , j ) , end = " ▁ " ) NEW_LINE DEDENT
for j in range ( 1 , i + 1 ) : NEW_LINE INDENT print ( TrinomialValue ( i , j ) , end = " ▁ " ) NEW_LINE DEDENT print ( " " , end = ' ' ) NEW_LINE
n = 4 NEW_LINE printTrinomial ( n ) NEW_LINE
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
def sumOfLargePrimeFactor ( n ) : NEW_LINE
prime = [ 0 ] * ( n + 1 ) NEW_LINE sum = 0 NEW_LINE max = int ( n / 2 ) NEW_LINE for p in range ( 2 , max + 1 ) : NEW_LINE
if prime [ p ] == 0 : NEW_LINE
for i in range ( p * 2 , n + 1 , p ) : NEW_LINE INDENT prime [ i ] = p NEW_LINE DEDENT
for p in range ( 2 , n + 1 ) : NEW_LINE
if prime [ p ] : NEW_LINE INDENT sum += prime [ p ] NEW_LINE DEDENT
else : NEW_LINE INDENT sum += p NEW_LINE DEDENT
return sum NEW_LINE
n = 12 NEW_LINE print ( " Sum ▁ = " , sumOfLargePrimeFactor ( n ) ) NEW_LINE
def calculate_sum ( a , N ) : NEW_LINE
m = N / a NEW_LINE
sum = m * ( m + 1 ) / 2 NEW_LINE
ans = a * sum NEW_LINE print ( " Sum ▁ of ▁ multiples ▁ of ▁ " , a , " ▁ up ▁ to ▁ " , N , " ▁ = ▁ " , ans ) NEW_LINE
calculate_sum ( 7 , 49 ) NEW_LINE
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
def ispowerof2 ( num ) : NEW_LINE INDENT if ( ( num & ( num - 1 ) ) == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT return 0 NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT num = 549755813888 NEW_LINE print ( ispowerof2 ( num ) ) NEW_LINE DEDENT
def counDivisors ( X ) : NEW_LINE
count = 0 NEW_LINE
for i in range ( 1 , X + 1 ) : NEW_LINE INDENT if ( X % i == 0 ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT DEDENT
return count NEW_LINE
def countDivisorsMult ( arr , n ) : NEW_LINE
mul = 1 NEW_LINE for i in range ( n ) : NEW_LINE INDENT mul *= arr [ i ] NEW_LINE DEDENT
return counDivisors ( mul ) NEW_LINE
arr = [ 2 , 4 , 6 ] NEW_LINE n = len ( arr ) NEW_LINE print ( countDivisorsMult ( arr , n ) ) NEW_LINE
from collections import defaultdict NEW_LINE def SieveOfEratosthenes ( largest , prime ) : NEW_LINE
isPrime = [ True ] * ( largest + 1 ) NEW_LINE p = 2 NEW_LINE while p * p <= largest : NEW_LINE
if ( isPrime [ p ] == True ) : NEW_LINE
for i in range ( p * 2 , largest + 1 , p ) : NEW_LINE INDENT isPrime [ i ] = False NEW_LINE DEDENT p += 1 NEW_LINE
for p in range ( 2 , largest + 1 ) : NEW_LINE INDENT if ( isPrime [ p ] ) : NEW_LINE INDENT prime . append ( p ) NEW_LINE DEDENT DEDENT
def countDivisorsMult ( arr , n ) : NEW_LINE
largest = max ( arr ) NEW_LINE prime = [ ] NEW_LINE SieveOfEratosthenes ( largest , prime ) NEW_LINE
mp = defaultdict ( int ) NEW_LINE for i in range ( n ) : NEW_LINE INDENT for j in range ( len ( prime ) ) : NEW_LINE INDENT while ( arr [ i ] > 1 and arr [ i ] % prime [ j ] == 0 ) : NEW_LINE INDENT arr [ i ] //= prime [ j ] NEW_LINE mp [ prime [ j ] ] += 1 NEW_LINE DEDENT DEDENT if ( arr [ i ] != 1 ) : NEW_LINE INDENT mp [ arr [ i ] ] += 1 NEW_LINE DEDENT DEDENT
res = 1 NEW_LINE for it in mp . values ( ) : NEW_LINE INDENT res *= ( it + 1 ) NEW_LINE DEDENT return res NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 2 , 4 , 6 ] NEW_LINE n = len ( arr ) NEW_LINE print ( countDivisorsMult ( arr , n ) ) NEW_LINE DEDENT
from math import sqrt NEW_LINE
def findPrimeNos ( L , R , M ) : NEW_LINE
for i in range ( L , R + 1 ) : NEW_LINE INDENT M [ i ] = M . get ( i , 0 ) + 1 NEW_LINE DEDENT
if ( 1 in M ) : NEW_LINE INDENT M . pop ( 1 ) NEW_LINE DEDENT
for i in range ( 2 , int ( sqrt ( R ) ) + 1 , 1 ) : NEW_LINE INDENT multiple = 2 NEW_LINE while ( ( i * multiple ) <= R ) : NEW_LINE DEDENT
if ( ( i * multiple ) in M ) : NEW_LINE
M . pop ( i * multiple ) NEW_LINE
multiple += 1 NEW_LINE
def getPrimePairs ( L , R , K ) : NEW_LINE INDENT M = { } NEW_LINE DEDENT
findPrimeNos ( L , R , M ) NEW_LINE
for key , values in M . items ( ) : NEW_LINE
if ( ( key + K ) in M ) : NEW_LINE INDENT print ( " ( " , key , " , " , key + K , " ) " , end = " ▁ " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE
L = 1 NEW_LINE R = 19 NEW_LINE
K = 6 NEW_LINE
getPrimePairs ( L , R , K ) NEW_LINE
def EnneacontahexagonNum ( n ) : NEW_LINE INDENT return ( 94 * n * n - 92 * n ) // 2 ; NEW_LINE DEDENT
n = 3 ; NEW_LINE print ( EnneacontahexagonNum ( n ) ) ; NEW_LINE
def find_composite_nos ( n ) : NEW_LINE INDENT print ( 9 * n , 8 * n ) ; NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 4 ; NEW_LINE find_composite_nos ( n ) ; NEW_LINE DEDENT
def freqPairs ( arr , n ) : NEW_LINE
max = arr [ 0 ] NEW_LINE for i in range ( len ( arr ) ) : NEW_LINE INDENT if arr [ i ] > max : NEW_LINE INDENT max = arr [ i ] NEW_LINE DEDENT DEDENT
freq = [ 0 for i in range ( max + 1 ) ] NEW_LINE
count = 0 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT freq [ arr [ i ] ] += 1 NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT for j in range ( 2 * arr [ i ] , max + 1 , arr [ i ] ) : NEW_LINE DEDENT
if ( freq [ j ] >= 1 ) : NEW_LINE INDENT count += freq [ j ] NEW_LINE DEDENT
if ( freq [ arr [ i ] ] > 1 ) : NEW_LINE INDENT count += freq [ arr [ i ] ] - 1 NEW_LINE freq [ arr [ i ] ] -= 1 NEW_LINE DEDENT return count NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 3 , 2 , 4 , 2 , 6 ] NEW_LINE n = len ( arr ) NEW_LINE print ( freqPairs ( arr , n ) ) NEW_LINE DEDENT
def Nth_Term ( n ) : NEW_LINE INDENT return ( 2 * pow ( n , 3 ) - 3 * pow ( n , 2 ) + n + 6 ) // 6 NEW_LINE DEDENT
N = 8 NEW_LINE print ( Nth_Term ( N ) ) NEW_LINE
def printNthElement ( n ) : NEW_LINE
arr = [ 0 ] * ( n + 1 ) ; NEW_LINE arr [ 1 ] = 3 NEW_LINE arr [ 2 ] = 5 NEW_LINE for i in range ( 3 , n + 1 ) : NEW_LINE
if ( i % 2 != 0 ) : NEW_LINE INDENT arr [ i ] = arr [ i // 2 ] * 10 + 3 NEW_LINE DEDENT else : NEW_LINE INDENT arr [ i ] = arr [ ( i // 2 ) - 1 ] * 10 + 5 NEW_LINE DEDENT return arr [ n ] NEW_LINE
n = 6 NEW_LINE print ( printNthElement ( n ) ) NEW_LINE
def nthTerm ( N ) : NEW_LINE
return ( N * ( ( N // 2 ) + ( ( N % 2 ) * 2 ) + N ) ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 5 NEW_LINE DEDENT
print ( " Nth ▁ term ▁ for ▁ N ▁ = ▁ " , N , " ▁ : ▁ " , nthTerm ( N ) ) NEW_LINE
def series ( A , X , n ) : NEW_LINE
term = pow ( A , n ) NEW_LINE print ( term , end = " ▁ " ) NEW_LINE
for i in range ( 1 , n + 1 ) : NEW_LINE
term = int ( term * X * ( n - i + 1 ) / ( i * A ) ) NEW_LINE print ( term , end = " ▁ " ) NEW_LINE
A = 3 ; X = 4 ; n = 5 NEW_LINE series ( A , X , n ) NEW_LINE
import math NEW_LINE
def Div_by_8 ( n ) : NEW_LINE INDENT return ( ( ( n >> 3 ) << 3 ) == n ) NEW_LINE DEDENT
n = 16 NEW_LINE if ( Div_by_8 ( n ) ) : NEW_LINE INDENT print ( " YES " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NO " ) NEW_LINE DEDENT
def averageEven ( n ) : NEW_LINE INDENT if ( n % 2 != 0 ) : NEW_LINE INDENT print ( " Invalid ▁ Input " ) NEW_LINE return - 1 NEW_LINE DEDENT sm = 0 NEW_LINE count = 0 NEW_LINE while ( n >= 2 ) : NEW_LINE DEDENT
count = count + 1 NEW_LINE
sm = sm + n NEW_LINE n = n - 2 NEW_LINE return sm // count NEW_LINE
n = 16 NEW_LINE print ( averageEven ( n ) ) NEW_LINE
def averageEven ( n ) : NEW_LINE INDENT if ( n % 2 != 0 ) : NEW_LINE INDENT print ( " Invalid ▁ Input " ) NEW_LINE return - 1 NEW_LINE DEDENT return ( n + 2 ) // 2 NEW_LINE DEDENT
n = 16 NEW_LINE print ( averageEven ( n ) ) NEW_LINE
def gcd ( a , b ) : NEW_LINE
if a == 0 or b == 0 : NEW_LINE INDENT return 0 NEW_LINE DEDENT
if a == b : NEW_LINE INDENT return a NEW_LINE DEDENT
if a > b : NEW_LINE INDENT return gcd ( a - b , b ) NEW_LINE DEDENT return gcd ( a , b - a ) NEW_LINE
def cpFact ( x , y ) : NEW_LINE INDENT while gcd ( x , y ) != 1 : NEW_LINE INDENT x = x / gcd ( x , y ) NEW_LINE DEDENT return int ( x ) NEW_LINE DEDENT
x = 15 NEW_LINE y = 3 NEW_LINE print ( cpFact ( x , y ) ) NEW_LINE x = 14 NEW_LINE y = 28 NEW_LINE print ( cpFact ( x , y ) ) NEW_LINE x = 7 NEW_LINE y = 3 NEW_LINE print ( cpFact ( x , y ) ) NEW_LINE
def counLastDigitK ( low , high , k ) : NEW_LINE INDENT count = 0 NEW_LINE for i in range ( low , high + 1 ) : NEW_LINE INDENT if ( i % 10 == k ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT DEDENT return count NEW_LINE DEDENT
low = 3 NEW_LINE high = 35 NEW_LINE k = 3 NEW_LINE print ( counLastDigitK ( low , high , k ) ) NEW_LINE
import math NEW_LINE def printTaxicab2 ( N ) : NEW_LINE
i , count = 1 , 0 NEW_LINE while ( count < N ) : NEW_LINE INDENT int_count = 0 NEW_LINE DEDENT
for j in range ( 1 , math . ceil ( pow ( i , 1.0 / 3 ) ) + 1 ) : NEW_LINE INDENT for k in range ( j + 1 , math . ceil ( pow ( i , 1.0 / 3 ) ) + 1 ) : NEW_LINE INDENT if ( j * j * j + k * k * k == i ) : NEW_LINE INDENT int_count += 1 NEW_LINE DEDENT DEDENT DEDENT
if ( int_count == 2 ) : NEW_LINE INDENT count += 1 NEW_LINE print ( count , " ▁ " , i ) NEW_LINE DEDENT i += 1 NEW_LINE
N = 5 NEW_LINE printTaxicab2 ( N ) NEW_LINE
def isComposite ( n ) : NEW_LINE
if ( n <= 1 ) : NEW_LINE INDENT return False NEW_LINE DEDENT if ( n <= 3 ) : NEW_LINE INDENT return False NEW_LINE DEDENT
if ( n % 2 == 0 or n % 3 == 0 ) : NEW_LINE INDENT return True NEW_LINE DEDENT i = 5 NEW_LINE while ( i * i <= n ) : NEW_LINE INDENT if ( n % i == 0 or n % ( i + 2 ) == 0 ) : NEW_LINE INDENT return True NEW_LINE DEDENT i = i + 6 NEW_LINE DEDENT return False NEW_LINE
print ( " true " ) if ( isComposite ( 11 ) ) else print ( " false " ) NEW_LINE print ( " true " ) if ( isComposite ( 15 ) ) else print ( " false " ) NEW_LINE
def isPrime ( n ) : NEW_LINE
if n <= 1 : NEW_LINE INDENT return False NEW_LINE DEDENT
for i in range ( 2 , n ) : NEW_LINE INDENT if n % i == 0 : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT return True NEW_LINE
def findPrime ( n ) : NEW_LINE INDENT num = n + 1 NEW_LINE DEDENT
while ( num ) : NEW_LINE
if isPrime ( num ) : NEW_LINE INDENT return num NEW_LINE DEDENT
num += 1 NEW_LINE return 0 NEW_LINE
def minNumber ( arr ) : NEW_LINE INDENT s = 0 NEW_LINE DEDENT
for i in range ( 0 , len ( arr ) ) : NEW_LINE INDENT s += arr [ i ] NEW_LINE DEDENT
if isPrime ( s ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
num = findPrime ( s ) NEW_LINE
return num - s NEW_LINE
arr = [ 2 , 4 , 6 , 8 , 12 ] NEW_LINE print ( minNumber ( arr ) ) NEW_LINE
def fact ( n ) : NEW_LINE INDENT if ( n == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT return n * fact ( n - 1 ) NEW_LINE DEDENT
def div ( x ) : NEW_LINE INDENT ans = 0 ; NEW_LINE for i in range ( 1 , x + 1 ) : NEW_LINE INDENT if ( x % i == 0 ) : NEW_LINE INDENT ans += i NEW_LINE DEDENT DEDENT return ans NEW_LINE DEDENT
def sumFactDiv ( n ) : NEW_LINE INDENT return div ( fact ( n ) ) NEW_LINE DEDENT
n = 4 NEW_LINE print ( sumFactDiv ( n ) ) NEW_LINE
allPrimes = [ ] ; NEW_LINE
def sieve ( n ) : NEW_LINE
prime = [ True ] * ( n + 1 ) ; NEW_LINE
p = 2 ; NEW_LINE while ( p * p <= n ) : NEW_LINE
if ( prime [ p ] == True ) : NEW_LINE
for i in range ( p * 2 , n + 1 , p ) : NEW_LINE INDENT prime [ i ] = False ; NEW_LINE DEDENT p += 1 ; NEW_LINE
for p in range ( 2 , n + 1 ) : NEW_LINE INDENT if ( prime [ p ] ) : NEW_LINE INDENT allPrimes . append ( p ) ; NEW_LINE DEDENT DEDENT
def factorialDivisors ( n ) : NEW_LINE
result = 1 ; NEW_LINE
for i in range ( len ( allPrimes ) ) : NEW_LINE
p = allPrimes [ i ] ; NEW_LINE
exp = 0 ; NEW_LINE while ( p <= n ) : NEW_LINE INDENT exp = exp + int ( n / p ) ; NEW_LINE p = p * allPrimes [ i ] ; NEW_LINE DEDENT
result = int ( result * ( pow ( allPrimes [ i ] , exp + 1 ) - 1 ) / ( allPrimes [ i ] - 1 ) ) ; NEW_LINE
return result ; NEW_LINE
print ( factorialDivisors ( 4 ) ) ; NEW_LINE
def checkPandigital ( b , n ) : NEW_LINE
if ( len ( n ) < b ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT hash = [ 0 ] * b ; NEW_LINE
for i in range ( len ( n ) ) : NEW_LINE
if ( n [ i ] >= '0' and n [ i ] <= '9' ) : NEW_LINE INDENT hash [ ord ( n [ i ] ) - ord ( '0' ) ] = 1 ; NEW_LINE DEDENT
elif ( ord ( n [ i ] ) - ord ( ' A ' ) <= b - 11 ) : NEW_LINE INDENT hash [ ord ( n [ i ] ) - ord ( ' A ' ) + 10 ] = 1 ; NEW_LINE DEDENT
for i in range ( b ) : NEW_LINE INDENT if ( hash [ i ] == 0 ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT DEDENT return 1 ; NEW_LINE
b = 13 ; NEW_LINE n = "1298450376ABC " ; NEW_LINE if ( checkPandigital ( b , n ) ) : NEW_LINE INDENT print ( " Yes " ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) ; NEW_LINE DEDENT
def conver ( m , n ) : NEW_LINE INDENT if ( m == n ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT DEDENT
if ( m > n ) : NEW_LINE INDENT return m - n NEW_LINE DEDENT
if ( m <= 0 and n > 0 ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
if ( n % 2 == 1 ) : NEW_LINE
return 1 + conver ( m , n + 1 ) NEW_LINE
else : NEW_LINE
return 1 + conver ( m , n / 2 ) NEW_LINE
m = 3 NEW_LINE n = 11 NEW_LINE print ( " Minimum ▁ number ▁ of ▁ operations ▁ : " , conver ( m , n ) ) NEW_LINE
MAX = 10000 ; NEW_LINE prodDig = [ 0 ] * MAX ; NEW_LINE
def getDigitProduct ( x ) : NEW_LINE
if ( x < 10 ) : NEW_LINE INDENT return x ; NEW_LINE DEDENT
if ( prodDig [ x ] != 0 ) : NEW_LINE INDENT return prodDig [ x ] ; NEW_LINE DEDENT
prod = ( int ( x % 10 ) * getDigitProduct ( int ( x / 10 ) ) ) ; NEW_LINE prodDig [ x ] = prod ; NEW_LINE return prod ; NEW_LINE
def findSeed ( n ) : NEW_LINE
res = [ ] ; NEW_LINE for i in range ( 1 , int ( n / 2 + 2 ) ) : NEW_LINE INDENT if ( i * getDigitProduct ( i ) == n ) : NEW_LINE INDENT res . append ( i ) ; NEW_LINE DEDENT DEDENT
if ( len ( res ) == 0 ) : NEW_LINE INDENT print ( " NO ▁ seed ▁ exists " ) ; NEW_LINE return ; NEW_LINE DEDENT
for i in range ( len ( res ) ) : NEW_LINE INDENT print ( res [ i ] , end = " ▁ " ) ; NEW_LINE DEDENT
n = 138 ; NEW_LINE findSeed ( n ) ; NEW_LINE
from math import sqrt NEW_LINE
def maxPrimefactorNum ( N ) : NEW_LINE INDENT arr = [ 0 for i in range ( N + 5 ) ] NEW_LINE DEDENT
for i in range ( 2 , int ( sqrt ( N ) ) + 1 , 1 ) : NEW_LINE INDENT if ( arr [ i ] == 0 ) : NEW_LINE INDENT for j in range ( 2 * i , N + 1 , i ) : NEW_LINE INDENT arr [ j ] += 1 NEW_LINE DEDENT DEDENT arr [ i ] = 1 NEW_LINE DEDENT maxval = 0 NEW_LINE maxint = 1 NEW_LINE
for i in range ( 1 , N + 1 , 1 ) : NEW_LINE INDENT if ( arr [ i ] > maxval ) : NEW_LINE INDENT maxval = arr [ i ] NEW_LINE maxint = i NEW_LINE DEDENT DEDENT return maxint NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 40 NEW_LINE print ( maxPrimefactorNum ( N ) ) NEW_LINE DEDENT
def SubArraySum ( arr , n ) : NEW_LINE INDENT result = 0 NEW_LINE DEDENT
for i in range ( 0 , n ) : NEW_LINE INDENT result += ( arr [ i ] * ( i + 1 ) * ( n - i ) ) NEW_LINE DEDENT
return result NEW_LINE
arr = [ 1 , 2 , 3 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Sum ▁ of ▁ SubArray ▁ : ▁ " , SubArraySum ( arr , n ) ) NEW_LINE
def highestPowerof2 ( n ) : NEW_LINE INDENT res = 0 ; NEW_LINE for i in range ( n , 0 , - 1 ) : NEW_LINE DEDENT
if ( ( i & ( i - 1 ) ) == 0 ) : NEW_LINE INDENT res = i ; NEW_LINE break ; NEW_LINE DEDENT return res ; NEW_LINE
n = 10 ; NEW_LINE print ( highestPowerof2 ( n ) ) ; NEW_LINE
import math NEW_LINE
def findPairs ( n ) : NEW_LINE
cubeRoot = int ( math . pow ( n , 1.0 / 3.0 ) ) ; NEW_LINE
' NEW_LINE INDENT cube = [ 0 ] * ( cubeRoot + 1 ) ; NEW_LINE DEDENT
for i in range ( 1 , cubeRoot + 1 ) : NEW_LINE INDENT cube [ i ] = i * i * i ; NEW_LINE DEDENT
l = 1 ; NEW_LINE r = cubeRoot ; NEW_LINE while ( l < r ) : NEW_LINE INDENT if ( cube [ l ] + cube [ r ] < n ) : NEW_LINE INDENT l += 1 ; NEW_LINE DEDENT elif ( cube [ l ] + cube [ r ] > n ) : NEW_LINE INDENT r -= 1 ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " ( " , l , " , ▁ " , math . floor ( r ) , " ) " , end = " " ) ; NEW_LINE print ( ) ; NEW_LINE l += 1 ; NEW_LINE r -= 1 ; NEW_LINE DEDENT DEDENT
n = 20683 ; NEW_LINE findPairs ( n ) ; NEW_LINE
def findPairs ( n ) : NEW_LINE
cubeRoot = pow ( n , 1.0 / 3.0 ) ; NEW_LINE
s = { } NEW_LINE
for x in range ( int ( cubeRoot ) ) : NEW_LINE INDENT for y in range ( x + 1 , int ( cubeRoot ) + 1 ) : NEW_LINE DEDENT
sum = x * x * x + y * y * y ; NEW_LINE
if ( sum != n ) : NEW_LINE INDENT continue ; NEW_LINE DEDENT
if sum in s . keys ( ) : NEW_LINE INDENT print ( " ( " + str ( s [ sum ] [ 0 ] ) + " , ▁ " + str ( s [ sum ] [ 1 ] ) + " ) ▁ and ▁ ( " + str ( x ) + " , ▁ " + str ( y ) + " ) " +   " " ) NEW_LINE DEDENT else : NEW_LINE
s [ sum ] = [ x , y ] NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 13832 NEW_LINE findPairs ( n ) NEW_LINE DEDENT
import math as mt NEW_LINE
def gcd ( a , b ) : NEW_LINE INDENT while ( b != 0 ) : NEW_LINE INDENT t = b NEW_LINE b = a % b NEW_LINE a = t NEW_LINE DEDENT return a NEW_LINE DEDENT
def findMinDiff ( a , b , x , y ) : NEW_LINE
g = gcd ( a , b ) NEW_LINE
diff = abs ( x - y ) % g NEW_LINE return min ( diff , g - diff ) NEW_LINE
a , b , x , y = 20 , 52 , 5 , 7 NEW_LINE print ( findMinDiff ( a , b , x , y ) ) NEW_LINE
import math NEW_LINE
def printDivisors ( n ) : NEW_LINE INDENT list = [ ] NEW_LINE DEDENT
for i in range ( 1 , int ( math . sqrt ( n ) + 1 ) ) : NEW_LINE INDENT if ( n % i == 0 ) : NEW_LINE DEDENT
if ( n / i == i ) : NEW_LINE INDENT print ( i , end = " ▁ " ) NEW_LINE DEDENT else : NEW_LINE
print ( i , end = " ▁ " ) NEW_LINE list . append ( int ( n / i ) ) NEW_LINE
for i in list [ : : - 1 ] : NEW_LINE INDENT print ( i , end = " ▁ " ) NEW_LINE DEDENT
print ( " The ▁ divisors ▁ of ▁ 100 ▁ are : ▁ " ) NEW_LINE printDivisors ( 100 ) NEW_LINE
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
def SieveOfAtkin ( limit ) : NEW_LINE
if ( limit > 2 ) : NEW_LINE INDENT print ( 2 , end = " ▁ " ) NEW_LINE DEDENT if ( limit > 3 ) : NEW_LINE INDENT print ( 3 , end = " ▁ " ) NEW_LINE DEDENT
sieve = [ False ] * limit NEW_LINE for i in range ( 0 , limit ) : NEW_LINE INDENT sieve [ i ] = False NEW_LINE DEDENT
x = 1 NEW_LINE while ( x * x < limit ) : NEW_LINE INDENT y = 1 NEW_LINE while ( y * y < limit ) : NEW_LINE DEDENT
n = ( 4 * x * x ) + ( y * y ) NEW_LINE if ( n <= limit and ( n % 12 == 1 or n % 12 == 5 ) ) : NEW_LINE INDENT sieve [ n ] ^= True NEW_LINE DEDENT n = ( 3 * x * x ) + ( y * y ) NEW_LINE if ( n <= limit and n % 12 == 7 ) : NEW_LINE INDENT sieve [ n ] ^= True NEW_LINE DEDENT n = ( 3 * x * x ) - ( y * y ) NEW_LINE if ( x > y and n <= limit and n % 12 == 11 ) : NEW_LINE INDENT sieve [ n ] ^= True NEW_LINE DEDENT y += 1 NEW_LINE x += 1 NEW_LINE
r = 5 NEW_LINE while ( r * r < limit ) : NEW_LINE INDENT if ( sieve [ r ] ) : NEW_LINE INDENT for i in range ( r * r , limit , r * r ) : NEW_LINE INDENT sieve [ i ] = False NEW_LINE DEDENT DEDENT DEDENT
for a in range ( 5 , limit ) : NEW_LINE INDENT if ( sieve [ a ] ) : NEW_LINE INDENT print ( a , end = " ▁ " ) NEW_LINE DEDENT DEDENT
limit = 20 NEW_LINE SieveOfAtkin ( limit ) NEW_LINE
def isInside ( circle_x , circle_y , rad , x , y ) : NEW_LINE
if ( ( x - circle_x ) * ( x - circle_x ) + ( y - circle_y ) * ( y - circle_y ) <= rad * rad ) : NEW_LINE INDENT return True ; NEW_LINE DEDENT else : NEW_LINE INDENT return False ; NEW_LINE DEDENT
x = 1 ; NEW_LINE y = 1 ; NEW_LINE circle_x = 0 ; NEW_LINE circle_y = 1 ; NEW_LINE rad = 2 ; NEW_LINE if ( isInside ( circle_x , circle_y , rad , x , y ) ) : NEW_LINE INDENT print ( " Inside " ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Outside " ) ; NEW_LINE DEDENT
def eval ( a , op , b ) : NEW_LINE INDENT if op == ' + ' : return a + b NEW_LINE if op == ' - ' : return a - b NEW_LINE if op == ' * ' : return a * b NEW_LINE DEDENT
def evaluateAll ( expr , low , high ) : NEW_LINE
res = [ ] NEW_LINE
if low == high : NEW_LINE INDENT res . append ( int ( expr [ low ] ) ) NEW_LINE return res NEW_LINE DEDENT
if low == ( high - 2 ) : NEW_LINE INDENT num = eval ( int ( expr [ low ] ) , expr [ low + 1 ] , int ( expr [ low + 2 ] ) ) NEW_LINE res . append ( num ) NEW_LINE return res NEW_LINE DEDENT
for i in range ( low + 1 , high + 1 , 2 ) : NEW_LINE
' NEW_LINE INDENT l = evaluateAll ( expr , low , i - 1 ) NEW_LINE DEDENT
' NEW_LINE INDENT r = evaluateAll ( expr , i + 1 , high ) NEW_LINE DEDENT
' NEW_LINE INDENT for s1 in range ( 0 , len ( l ) ) : NEW_LINE DEDENT
' NEW_LINE INDENT for s2 in range ( 0 , len ( r ) ) : NEW_LINE DEDENT
val = eval ( l [ s1 ] , expr [ i ] , r [ s2 ] ) NEW_LINE res . append ( val ) NEW_LINE return res NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT expr = "1*2 + 3*4" NEW_LINE length = len ( expr ) NEW_LINE ans = evaluateAll ( expr , 0 , length - 1 ) NEW_LINE for i in range ( 0 , len ( ans ) ) : NEW_LINE INDENT print ( ans [ i ] ) NEW_LINE DEDENT DEDENT
import math NEW_LINE
def isLucky ( n ) : NEW_LINE
ar = [ 0 ] * 10 NEW_LINE
while ( n > 0 ) : NEW_LINE
digit = math . floor ( n % 10 ) NEW_LINE
if ( ar [ digit ] ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
ar [ digit ] = 1 NEW_LINE
n = n / 10 NEW_LINE return 1 NEW_LINE
arr = [ 1291 , 897 , 4566 , 1232 , 80 , 700 ] NEW_LINE n = len ( arr ) NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT k = arr [ i ] NEW_LINE if ( isLucky ( k ) ) : NEW_LINE INDENT print ( k , " ▁ is ▁ Lucky ▁ " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( k , " ▁ is ▁ not ▁ Lucky ▁ " ) NEW_LINE DEDENT DEDENT
def printSquares ( n ) : NEW_LINE
square = 0 NEW_LINE odd = 1 NEW_LINE
for x in range ( 0 , n ) : NEW_LINE
print ( square , end = " ▁ " ) NEW_LINE
' NEW_LINE INDENT square = square + odd NEW_LINE odd = odd + 2 NEW_LINE DEDENT
n = 5 ; NEW_LINE printSquares ( n ) NEW_LINE
def reversDigits ( num ) : NEW_LINE INDENT global rev_num NEW_LINE global base_pos NEW_LINE if ( num > 0 ) : NEW_LINE INDENT reversDigits ( ( int ) ( num / 10 ) ) NEW_LINE rev_num += ( num % 10 ) * base_pos NEW_LINE base_pos *= 10 NEW_LINE DEDENT return rev_num NEW_LINE DEDENT
num = 4562 NEW_LINE print ( " Reverse ▁ of ▁ no . ▁ is ▁ " , reversDigits ( num ) ) NEW_LINE
def RecursiveFunction ( ref , bit ) : NEW_LINE
if ( len ( ref ) == 0 or bit < 0 ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT curr_on = [ ] NEW_LINE curr_off = [ ] NEW_LINE for i in range ( len ( ref ) ) : NEW_LINE
if ( ( ( ref [ i ] >> bit ) & 1 ) == 0 ) : NEW_LINE INDENT curr_off . append ( ref [ i ] ) NEW_LINE DEDENT
else : NEW_LINE INDENT curr_on . append ( ref [ i ] ) NEW_LINE DEDENT
if ( len ( curr_off ) == 0 ) : NEW_LINE INDENT return RecursiveFunction ( curr_on , bit - 1 ) NEW_LINE DEDENT
if ( len ( curr_on ) == 0 ) : NEW_LINE INDENT return RecursiveFunction ( curr_off , bit - 1 ) NEW_LINE DEDENT
return ( min ( RecursiveFunction ( curr_off , bit - 1 ) , RecursiveFunction ( curr_on , bit - 1 ) ) + ( 1 << bit ) ) NEW_LINE
def PrintMinimum ( a , n ) : NEW_LINE INDENT v = [ ] NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT v . append ( a [ i ] ) NEW_LINE DEDENT
print ( RecursiveFunction ( v , 30 ) ) NEW_LINE
arr = [ 3 , 2 , 1 ] NEW_LINE size = len ( arr ) NEW_LINE PrintMinimum ( arr , size ) NEW_LINE
def cntElements ( arr , n ) : NEW_LINE
cnt = 0 NEW_LINE
for i in range ( n - 2 ) : NEW_LINE
if ( arr [ i ] == ( arr [ i + 1 ] ^ arr [ i + 2 ] ) ) : NEW_LINE INDENT cnt += 1 NEW_LINE DEDENT return cnt NEW_LINE
arr = [ 4 , 2 , 1 , 3 , 7 , 8 ] NEW_LINE n = len ( arr ) NEW_LINE print ( cntElements ( arr , n ) ) NEW_LINE
def xor_triplet ( arr , n ) : NEW_LINE
ans = 0 ; NEW_LINE
for i in range ( n ) : NEW_LINE
for j in range ( i + 1 , n ) : NEW_LINE
for k in range ( j , n ) : NEW_LINE INDENT xor1 = 0 ; xor2 = 0 ; NEW_LINE DEDENT
for x in range ( i , j ) : NEW_LINE INDENT xor1 ^= arr [ x ] ; NEW_LINE DEDENT
for x in range ( j , k + 1 ) : NEW_LINE INDENT xor2 ^= arr [ x ] ; NEW_LINE DEDENT
if ( xor1 == xor2 ) : NEW_LINE INDENT ans += 1 ; NEW_LINE DEDENT return ans ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 1 , 2 , 3 , 4 , 5 ] ; NEW_LINE n = len ( arr ) ; NEW_LINE DEDENT
print ( xor_triplet ( arr , n ) ) ; NEW_LINE
N = 100005 NEW_LINE Ideal_pair = 0 NEW_LINE
al = [ [ ] for i in range ( 100005 ) ] NEW_LINE bit = [ 0 for i in range ( N ) ] NEW_LINE root_node = [ 0 for i in range ( N ) ] NEW_LINE
def bit_q ( i , j ) : NEW_LINE INDENT sum = 0 NEW_LINE while ( j > 0 ) : NEW_LINE INDENT sum += bit [ j ] NEW_LINE j -= ( j & ( j * - 1 ) ) NEW_LINE DEDENT i -= 1 NEW_LINE while ( i > 0 ) : NEW_LINE INDENT sum -= bit [ i ] NEW_LINE i -= ( i & ( i * - 1 ) ) NEW_LINE DEDENT return sum NEW_LINE DEDENT
def bit_up ( i , diff ) : NEW_LINE INDENT while ( i <= n ) : NEW_LINE INDENT bit [ i ] += diff NEW_LINE i += i & - i NEW_LINE DEDENT DEDENT
def dfs ( node , x ) : NEW_LINE INDENT Ideal_pair = x NEW_LINE Ideal_pair += bit_q ( max ( 1 , node - k ) , min ( n , node + k ) ) NEW_LINE bit_up ( node , 1 ) NEW_LINE for i in range ( len ( al [ node ] ) ) : NEW_LINE INDENT Ideal_pair = dfs ( al [ node ] [ i ] , Ideal_pair ) NEW_LINE DEDENT bit_up ( node , - 1 ) NEW_LINE return Ideal_pair NEW_LINE DEDENT
def initialise ( ) : NEW_LINE INDENT Ideal_pair = 0 ; NEW_LINE for i in range ( n + 1 ) : NEW_LINE INDENT root_node [ i ] = True NEW_LINE bit [ i ] = 0 NEW_LINE DEDENT DEDENT
def Add_Edge ( x , y ) : NEW_LINE INDENT al [ x ] . append ( y ) NEW_LINE root_node [ y ] = False NEW_LINE DEDENT
def Idealpairs ( ) : NEW_LINE
r = - 1 NEW_LINE for i in range ( 1 , n + 1 , 1 ) : NEW_LINE INDENT if ( root_node [ i ] ) : NEW_LINE INDENT r = i NEW_LINE break NEW_LINE DEDENT DEDENT Ideal_pair = dfs ( r , 0 ) NEW_LINE return Ideal_pair NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 6 NEW_LINE k = 3 NEW_LINE initialise ( ) NEW_LINE DEDENT
Add_Edge ( 1 , 2 ) NEW_LINE Add_Edge ( 1 , 3 ) NEW_LINE Add_Edge ( 3 , 4 ) NEW_LINE Add_Edge ( 3 , 5 ) NEW_LINE Add_Edge ( 3 , 6 ) NEW_LINE
print ( Idealpairs ( ) ) NEW_LINE
def printSubsets ( n ) : NEW_LINE INDENT i = n NEW_LINE while ( i != 0 ) : NEW_LINE INDENT print ( i , end = " ▁ " ) NEW_LINE i = ( i - 1 ) & n NEW_LINE DEDENT print ( "0" ) NEW_LINE DEDENT
n = 9 NEW_LINE printSubsets ( n ) NEW_LINE
def isDivisibleby17 ( n ) : NEW_LINE
if ( n == 0 or n == 17 ) : NEW_LINE INDENT return True NEW_LINE DEDENT
if ( n < 17 ) : NEW_LINE INDENT return False NEW_LINE DEDENT
return isDivisibleby17 ( ( int ) ( n >> 4 ) - ( int ) ( n & 15 ) ) NEW_LINE
n = 35 NEW_LINE if ( isDivisibleby17 ( n ) ) : NEW_LINE INDENT print ( n , " is ▁ divisible ▁ by ▁ 17" ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( n , " is ▁ not ▁ divisible ▁ by ▁ 17" ) NEW_LINE DEDENT
import math NEW_LINE
def answer ( n ) : NEW_LINE
m = 2 ; NEW_LINE
ans = 1 ; NEW_LINE r = 1 ; NEW_LINE
while r < n : NEW_LINE
r = ( int ) ( ( pow ( 2 , m ) - 1 ) * ( pow ( 2 , m - 1 ) ) ) ; NEW_LINE
if r < n : NEW_LINE INDENT ans = r ; NEW_LINE DEDENT
m = m + 1 ; NEW_LINE return ans ; NEW_LINE
print ( answer ( 7 ) ) ; NEW_LINE
def setBitNumber ( n ) : NEW_LINE INDENT if ( n == 0 ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT msb = 0 ; NEW_LINE n = int ( n / 2 ) ; NEW_LINE while ( n > 0 ) : NEW_LINE INDENT n = int ( n / 2 ) ; NEW_LINE msb += 1 ; NEW_LINE DEDENT return ( 1 << msb ) ; NEW_LINE DEDENT
n = 0 ; NEW_LINE print ( setBitNumber ( n ) ) ; NEW_LINE
def setBitNumber ( n ) : NEW_LINE
n |= n >> 1 NEW_LINE
n |= n >> 2 NEW_LINE n |= n >> 4 NEW_LINE n |= n >> 8 NEW_LINE n |= n >> 16 NEW_LINE
n = n + 1 NEW_LINE
return ( n >> 1 ) NEW_LINE
n = 273 NEW_LINE print ( setBitNumber ( n ) ) NEW_LINE
def countTrailingZero ( x ) : NEW_LINE INDENT count = 0 NEW_LINE while ( ( x & 1 ) == 0 ) : NEW_LINE INDENT x = x >> 1 NEW_LINE count += 1 NEW_LINE DEDENT return count NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT print ( countTrailingZero ( 11 ) ) NEW_LINE DEDENT
def countTrailingZero ( x ) : NEW_LINE
lookup = [ 32 , 0 , 1 , 26 , 2 , 23 , 27 , 0 , 3 , 16 , 24 , 30 , 28 , 11 , 0 , 13 , 4 , 7 , 17 , 0 , 25 , 22 , 31 , 15 , 29 , 10 , 12 , 6 , 0 , 21 , 14 , 9 , 5 , 20 , 8 , 19 , 18 ] NEW_LINE
return lookup [ ( - x & x ) % 37 ] NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT print ( countTrailingZero ( 48 ) ) NEW_LINE DEDENT
def multiplyBySevenByEight ( n ) : NEW_LINE
' NEW_LINE INDENT return ( n - ( n >> 3 ) ) NEW_LINE DEDENT
n = 9 NEW_LINE print ( multiplyBySevenByEight ( n ) ) NEW_LINE
def multiplyBySevenByEight ( n ) : NEW_LINE
return ( ( n << 3 ) - n ) >> 3 ; NEW_LINE
n = 15 ; NEW_LINE print ( multiplyBySevenByEight ( n ) ) ; NEW_LINE
def countNumbers ( L , R , K ) : NEW_LINE
list = [ ] NEW_LINE
for i in range ( L , R + 1 ) : NEW_LINE
if ( isPalindrome ( i ) ) : NEW_LINE
list . append ( i ) NEW_LINE
count = 0 NEW_LINE
for i in range ( len ( list ) ) : NEW_LINE
right_index = search ( list , list [ i ] + K - 1 ) NEW_LINE
if ( right_index != - 1 ) : NEW_LINE INDENT count = max ( count , right_index - i + 1 ) NEW_LINE DEDENT
return count NEW_LINE
def search ( list , num ) : NEW_LINE INDENT low , high = 0 , len ( list ) - 1 NEW_LINE DEDENT
ans = - 1 NEW_LINE while ( low <= high ) : NEW_LINE
mid = low + ( high - low ) // 2 NEW_LINE
if ( list [ mid ] <= num ) : NEW_LINE
ans = mid NEW_LINE
low = mid + 1 NEW_LINE else : NEW_LINE
high = mid - 1 NEW_LINE
return ans NEW_LINE
def isPalindrome ( n ) : NEW_LINE INDENT rev = 0 NEW_LINE temp = n NEW_LINE DEDENT
while ( n > 0 ) : NEW_LINE INDENT rev = rev * 10 + n % 10 NEW_LINE n //= 10 NEW_LINE DEDENT
return rev == temp NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT L , R = 98 , 112 NEW_LINE K = 13 NEW_LINE print ( countNumbers ( L , R , K ) ) NEW_LINE DEDENT
def findMaximumSum ( a , n ) : NEW_LINE
prev_smaller = findPrevious ( a , n ) NEW_LINE
next_smaller = findNext ( a , n ) NEW_LINE max_value = 0 NEW_LINE for i in range ( n ) : NEW_LINE
max_value = max ( max_value , a [ i ] * ( next_smaller [ i ] - prev_smaller [ i ] - 1 ) ) NEW_LINE
return max_value NEW_LINE
def findPrevious ( a , n ) : NEW_LINE INDENT ps = [ 0 ] * n NEW_LINE DEDENT
ps [ 0 ] = - 1 NEW_LINE
stack = [ ] NEW_LINE
stack . append ( 0 ) NEW_LINE for i in range ( 1 , n ) : NEW_LINE
while len ( stack ) > 0 and a [ stack [ - 1 ] ] >= a [ i ] : NEW_LINE INDENT stack . pop ( ) NEW_LINE DEDENT
ps [ i ] = stack [ - 1 ] if len ( stack ) > 0 else - 1 NEW_LINE
stack . append ( i ) NEW_LINE
return ps NEW_LINE
def findNext ( a , n ) : NEW_LINE INDENT ns = [ 0 ] * n NEW_LINE ns [ n - 1 ] = n NEW_LINE DEDENT
stack = [ ] NEW_LINE stack . append ( n - 1 ) NEW_LINE
for i in range ( n - 2 , - 1 , - 1 ) : NEW_LINE
while ( len ( stack ) > 0 and a [ stack [ - 1 ] ] >= a [ i ] ) : NEW_LINE INDENT stack . pop ( ) NEW_LINE DEDENT
ns [ i ] = stack [ - 1 ] if len ( stack ) > 0 else n NEW_LINE
stack . append ( i ) NEW_LINE
return ns NEW_LINE
n = 3 NEW_LINE a = [ 80 , 48 , 82 ] NEW_LINE print ( findMaximumSum ( a , n ) ) NEW_LINE
MAX = 256 NEW_LINE def compare ( arr1 , arr2 ) : NEW_LINE INDENT global MAX NEW_LINE for i in range ( MAX ) : NEW_LINE INDENT if ( arr1 [ i ] != arr2 [ i ] ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT return True NEW_LINE DEDENT
def search ( pat , txt ) : NEW_LINE INDENT M = len ( pat ) NEW_LINE N = len ( txt ) NEW_LINE DEDENT
countP = [ 0 for i in range ( MAX ) ] NEW_LINE countTW = [ 0 for i in range ( MAX ) ] NEW_LINE for i in range ( M ) : NEW_LINE INDENT countP [ ord ( pat [ i ] ) ] += 1 NEW_LINE countTW [ ord ( txt [ i ] ) ] += 1 NEW_LINE DEDENT
for i in range ( M , N ) : NEW_LINE
if ( compare ( countP , countTW ) ) : NEW_LINE INDENT return True NEW_LINE DEDENT
countTW [ ord ( txt [ i ] ) ] += 1 NEW_LINE
countTW [ ord ( txt [ i - M ] ) ] -= 1 NEW_LINE
if ( compare ( countP , countTW ) ) : NEW_LINE INDENT return True NEW_LINE return False NEW_LINE DEDENT
txt = " BACDGABCDA " NEW_LINE pat = " ABCD " NEW_LINE if ( search ( pat , txt ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
def getMaxMedian ( arr , n , k ) : NEW_LINE INDENT size = n + k NEW_LINE DEDENT
arr . sort ( reverse = False ) NEW_LINE
if ( size % 2 == 0 ) : NEW_LINE INDENT median = ( arr [ int ( size / 2 ) - 1 ] + arr [ int ( size / 2 ) ] ) / 2 NEW_LINE return median NEW_LINE DEDENT
median = arr [ int ( size / 2 ) ] NEW_LINE return median NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 3 , 2 , 3 , 4 , 2 ] NEW_LINE n = len ( arr ) NEW_LINE k = 2 NEW_LINE print ( getMaxMedian ( arr , n , k ) ) NEW_LINE DEDENT
def printSorted ( a , b , c ) : NEW_LINE
get_max = max ( a , max ( b , c ) ) NEW_LINE
get_min = - max ( - a , max ( - b , - c ) ) NEW_LINE get_mid = ( a + b + c ) - ( get_max + get_min ) NEW_LINE print ( get_min , " ▁ " , get_mid , " ▁ " , get_max ) NEW_LINE
a , b , c = 4 , 1 , 9 NEW_LINE printSorted ( a , b , c ) NEW_LINE
def binarySearch ( a , item , low , high ) : NEW_LINE INDENT while ( low <= high ) : NEW_LINE INDENT mid = low + ( high - low ) // 2 NEW_LINE if ( item == a [ mid ] ) : NEW_LINE INDENT return mid + 1 NEW_LINE DEDENT elif ( item > a [ mid ] ) : NEW_LINE INDENT low = mid + 1 NEW_LINE DEDENT else : NEW_LINE INDENT high = mid - 1 NEW_LINE DEDENT DEDENT return low NEW_LINE DEDENT
' NEW_LINE def insertionSort ( a , n ) : NEW_LINE INDENT for i in range ( n ) : NEW_LINE INDENT j = i - 1 NEW_LINE selected = a [ i ] NEW_LINE DEDENT DEDENT
loc = binarySearch ( a , selected , 0 , j ) NEW_LINE
while ( j >= loc ) : NEW_LINE INDENT a [ j + 1 ] = a [ j ] NEW_LINE j -= 1 NEW_LINE DEDENT a [ j + 1 ] = selected NEW_LINE
a = [ 37 , 23 , 0 , 17 , 12 , 72 , 31 , 46 , 100 , 88 , 54 ] NEW_LINE n = len ( a ) NEW_LINE insertionSort ( a , n ) NEW_LINE print ( " Sorted ▁ array : ▁ " ) NEW_LINE for i in range ( n ) : NEW_LINE INDENT print ( a [ i ] , end = " ▁ " ) NEW_LINE DEDENT
def insertionSort ( arr ) : NEW_LINE INDENT for i in range ( 1 , len ( arr ) ) : NEW_LINE INDENT key = arr [ i ] NEW_LINE DEDENT DEDENT
j = i - 1 NEW_LINE while j >= 0 and key < arr [ j ] : NEW_LINE INDENT arr [ j + 1 ] = arr [ j ] NEW_LINE j -= 1 NEW_LINE DEDENT arr [ j + 1 ] = key NEW_LINE
arr = [ 12 , 11 , 13 , 5 , 6 ] NEW_LINE insertionSort ( arr ) NEW_LINE for i in range ( len ( arr ) ) : NEW_LINE INDENT print ( " % ▁ d " % arr [ i ] ) NEW_LINE DEDENT
def validPermutations ( str ) : NEW_LINE INDENT m = { } NEW_LINE DEDENT
count = len ( str ) NEW_LINE ans = 0 NEW_LINE
for i in range ( len ( str ) ) : NEW_LINE INDENT if ( str [ i ] in m ) : NEW_LINE INDENT m [ str [ i ] ] += 1 NEW_LINE DEDENT else : NEW_LINE INDENT m [ str [ i ] ] = 1 NEW_LINE DEDENT DEDENT for i in range ( len ( str ) ) : NEW_LINE
ans += count - m [ str [ i ] ] NEW_LINE
m [ str [ i ] ] -= 1 NEW_LINE count -= 1 NEW_LINE
return ans + 1 NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT str = " sstt " NEW_LINE print ( validPermutations ( str ) ) NEW_LINE DEDENT
def countPaths ( n , m ) : NEW_LINE
if ( n == 0 or m == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT
return ( countPaths ( n - 1 , m ) + countPaths ( n , m - 1 ) ) NEW_LINE
n = 3 NEW_LINE m = 2 NEW_LINE print ( " Number ▁ of ▁ Paths " , countPaths ( n , m ) ) NEW_LINE
def count ( S , m , n ) : NEW_LINE
if ( n == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT
if ( n < 0 ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT
if ( m <= 0 and n >= 1 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
return count ( S , m - 1 , n ) + count ( S , m , n - S [ m - 1 ] ) ; NEW_LINE
arr = [ 1 , 2 , 3 ] NEW_LINE m = len ( arr ) NEW_LINE print ( count ( arr , m , 4 ) ) NEW_LINE
def equalIgnoreCase ( str1 , str2 ) : NEW_LINE
str1 = str1 . upper ( ) ; NEW_LINE str2 = str2 . upper ( ) ; NEW_LINE
x = str1 == str2 ; NEW_LINE return x ; NEW_LINE
def equalIgnoreCaseUtil ( str1 , str2 ) : NEW_LINE INDENT res = equalIgnoreCase ( str1 , str2 ) ; NEW_LINE if ( res == True ) : NEW_LINE INDENT print ( " Same " ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Not ▁ Same " ) ; NEW_LINE DEDENT DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT str1 = " Geeks " ; NEW_LINE str2 = " geeks " ; NEW_LINE equalIgnoreCaseUtil ( str1 , str2 ) ; NEW_LINE str1 = " Geek " ; NEW_LINE str2 = " geeksforgeeks " ; NEW_LINE equalIgnoreCaseUtil ( str1 , str2 ) ; NEW_LINE DEDENT
def replaceConsonants ( string ) : NEW_LINE
res = " " ; NEW_LINE i = 0 ; count = 0 ; NEW_LINE
while ( i < len ( string ) ) : NEW_LINE
if ( string [ i ] != ' a ' and string [ i ] != ' e ' and string [ i ] != ' i ' and string [ i ] != ' o ' and string [ i ] != ' u ' ) : NEW_LINE INDENT i += 1 ; NEW_LINE count += 1 ; NEW_LINE DEDENT else : NEW_LINE
if ( count > 0 ) : NEW_LINE INDENT res += str ( count ) ; NEW_LINE DEDENT
res += string [ i ] ; NEW_LINE i += 1 NEW_LINE count = 0 ; NEW_LINE
if ( count > 0 ) : NEW_LINE INDENT res += str ( count ) ; NEW_LINE DEDENT
return res ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT string = " abcdeiop " ; NEW_LINE print ( replaceConsonants ( string ) ) ; NEW_LINE DEDENT
def isVowel ( c ) : NEW_LINE INDENT return ( c == ' a ' or c == ' e ' or c == ' i ' or c == ' o ' or c == ' u ' ) NEW_LINE DEDENT
def encryptString ( s , n , k ) : NEW_LINE INDENT countVowels = 0 NEW_LINE countConsonants = 0 NEW_LINE ans = " " NEW_LINE DEDENT
for l in range ( n - k + 1 ) : NEW_LINE INDENT countVowels = 0 NEW_LINE countConsonants = 0 NEW_LINE DEDENT
for r in range ( l , l + k ) : NEW_LINE
if ( isVowel ( s [ r ] ) == True ) : NEW_LINE INDENT countVowels += 1 NEW_LINE DEDENT else : NEW_LINE INDENT countConsonants += 1 NEW_LINE DEDENT
ans += ( str ) ( countVowels * countConsonants ) NEW_LINE return ans NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT s = " hello " NEW_LINE n = len ( s ) NEW_LINE k = 2 NEW_LINE print ( encryptString ( s , n , k ) ) NEW_LINE DEDENT
charBuffer = [ ] NEW_LINE def processWords ( input ) : NEW_LINE
s = input . split ( " ▁ " ) NEW_LINE for values in s : NEW_LINE
charBuffer . append ( values [ 0 ] ) NEW_LINE return charBuffer NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT input = " geeks ▁ for ▁ geeks " NEW_LINE print ( * processWords ( input ) , sep = " " ) NEW_LINE DEDENT
def generateAllStringsUtil ( K , str , n ) : NEW_LINE
if ( n == K ) : NEW_LINE
print ( * str [ : n ] , sep = " " , end = " ▁ " ) NEW_LINE return NEW_LINE
if ( str [ n - 1 ] == '1' ) : NEW_LINE INDENT str [ n ] = '0' NEW_LINE generateAllStringsUtil ( K , str , n + 1 ) NEW_LINE DEDENT
if ( str [ n - 1 ] == '0' ) : NEW_LINE INDENT str [ n ] = '0' NEW_LINE generateAllStringsUtil ( K , str , n + 1 ) NEW_LINE str [ n ] = '1' NEW_LINE generateAllStringsUtil ( K , str , n + 1 ) NEW_LINE DEDENT
def generateAllStrings ( K ) : NEW_LINE
if ( K <= 0 ) : NEW_LINE INDENT return NEW_LINE DEDENT
str = [ 0 ] * K NEW_LINE
' NEW_LINE INDENT str [ 0 ] = '0' NEW_LINE generateAllStringsUtil ( K , str , 1 ) NEW_LINE DEDENT
' NEW_LINE INDENT str [ 0 ] = '1' NEW_LINE generateAllStringsUtil ( K , str , 1 ) NEW_LINE DEDENT
K = 3 NEW_LINE generateAllStrings ( K ) NEW_LINE
def findVolume ( a ) : NEW_LINE
if ( a < 0 ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
r = a / 2 NEW_LINE
h = a NEW_LINE
V = 3.14 * pow ( r , 2 ) * h NEW_LINE return V NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = 5 NEW_LINE print ( findVolume ( a ) ) NEW_LINE DEDENT
def volumeTriangular ( a , b , h ) : NEW_LINE INDENT return ( 0.1666 ) * a * b * h NEW_LINE DEDENT
def volumeSquare ( b , h ) : NEW_LINE INDENT return ( 0.33 ) * b * b * h NEW_LINE DEDENT
def volumePentagonal ( a , b , h ) : NEW_LINE INDENT return ( 0.83 ) * a * b * h NEW_LINE DEDENT
def volumeHexagonal ( a , b , h ) : NEW_LINE INDENT return a * b * h NEW_LINE DEDENT
b = float ( 4 ) NEW_LINE h = float ( 9 ) NEW_LINE a = float ( 4 ) NEW_LINE print ( " Volume ▁ of ▁ triangular ▁ base ▁ pyramid ▁ is ▁ " , volumeTriangular ( a , b , h ) ) NEW_LINE print ( " Volume ▁ of ▁ square ▁ base ▁ pyramid ▁ is ▁ " , volumeSquare ( b , h ) ) NEW_LINE print ( " Volume ▁ of ▁ pentagonal ▁ base ▁ pyramid ▁ is ▁ " , volumePentagonal ( a , b , h ) ) NEW_LINE print ( " Volume ▁ of ▁ Hexagonal ▁ base ▁ pyramid ▁ is ▁ " , volumeHexagonal ( a , b , h ) ) NEW_LINE
def Area ( b1 , b2 , h ) : NEW_LINE INDENT return ( ( b1 + b2 ) / 2 ) * h NEW_LINE DEDENT
base1 = 8 ; base2 = 10 ; height = 6 NEW_LINE area = Area ( base1 , base2 , height ) NEW_LINE print ( " Area ▁ is : " , area ) NEW_LINE
def numberOfDiagonals ( n ) : NEW_LINE INDENT return n * ( n - 3 ) / 2 NEW_LINE DEDENT
def main ( ) : NEW_LINE INDENT n = 5 NEW_LINE print ( n , " ▁ sided ▁ convex ▁ polygon ▁ have ▁ " ) NEW_LINE print ( numberOfDiagonals ( n ) , " ▁ diagonals " ) NEW_LINE DEDENT if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT main ( ) NEW_LINE DEDENT
def maximumArea ( l , b , x , y ) : NEW_LINE
left , right , above , below = 0 , 0 , 0 , 0 NEW_LINE left = x * b NEW_LINE right = ( l - x - 1 ) * b NEW_LINE above = l * y NEW_LINE below = ( b - y - 1 ) * l NEW_LINE
print ( max ( max ( left , right ) , max ( above , below ) ) ) NEW_LINE
l = 8 NEW_LINE b = 8 NEW_LINE x = 0 NEW_LINE y = 0 NEW_LINE
maximumArea ( l , b , x , y ) NEW_LINE
def delCost ( s , cost ) : NEW_LINE
ans = 0 NEW_LINE
forMax = { } NEW_LINE
forTot = { } NEW_LINE
for i in range ( len ( s ) ) : NEW_LINE
if s [ i ] not in forMax : NEW_LINE INDENT forMax [ s [ i ] ] = cost [ i ] NEW_LINE DEDENT else : NEW_LINE
forMax [ s [ i ] ] = max ( cost [ i ] , forMax [ s [ i ] ] ) NEW_LINE
if s [ i ] not in forTot : NEW_LINE INDENT forTot [ s [ i ] ] = cost [ i ] NEW_LINE DEDENT else : NEW_LINE
forTot [ s [ i ] ] += cost [ i ] NEW_LINE
for i in forMax : NEW_LINE
ans += forTot [ i ] - forMax [ i ] NEW_LINE
return ans NEW_LINE
string = " AAABBB " NEW_LINE
cost = [ 1 , 2 , 3 , 4 , 5 , 6 ] NEW_LINE
print ( delCost ( string , cost ) ) NEW_LINE
MAX = 10000 NEW_LINE divisors = [ [ ] for i in range ( MAX + 1 ) ] NEW_LINE
def computeDivisors ( ) : NEW_LINE INDENT global divisors NEW_LINE global MAX NEW_LINE for i in range ( 1 , MAX + 1 , 1 ) : NEW_LINE INDENT for j in range ( i , MAX + 1 , i ) : NEW_LINE DEDENT DEDENT
divisors [ j ] . append ( i ) NEW_LINE
def getClosest ( val1 , val2 , target ) : NEW_LINE INDENT if ( target - val1 >= val2 - target ) : NEW_LINE INDENT return val2 NEW_LINE DEDENT else : NEW_LINE INDENT return val1 NEW_LINE DEDENT DEDENT
def findClosest ( arr , n , target ) : NEW_LINE
if ( target <= arr [ 0 ] ) : NEW_LINE INDENT return arr [ 0 ] NEW_LINE DEDENT if ( target >= arr [ n - 1 ] ) : NEW_LINE INDENT return arr [ n - 1 ] NEW_LINE DEDENT
i = 0 NEW_LINE j = n NEW_LINE mid = 0 NEW_LINE while ( i < j ) : NEW_LINE INDENT mid = ( i + j ) // 2 NEW_LINE if ( arr [ mid ] == target ) : NEW_LINE INDENT return arr [ mid ] NEW_LINE DEDENT DEDENT
if ( target < arr [ mid ] ) : NEW_LINE
if ( mid > 0 and target > arr [ mid - 1 ] ) : NEW_LINE INDENT return getClosest ( arr [ mid - 1 ] , arr [ mid ] , target ) NEW_LINE DEDENT
j = mid NEW_LINE
else : NEW_LINE INDENT if ( mid < n - 1 and target < arr [ mid + 1 ] ) : NEW_LINE INDENT return getClosest ( arr [ mid ] , arr [ mid + 1 ] , target ) NEW_LINE DEDENT DEDENT
i = mid + 1 NEW_LINE
return arr [ mid ] NEW_LINE
def printClosest ( N , X ) : NEW_LINE INDENT global divisors NEW_LINE DEDENT
computeDivisors ( ) NEW_LINE
ans = findClosest ( divisors [ N ] , len ( divisors [ N ] ) , X ) NEW_LINE
print ( ans ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
N = 16 NEW_LINE X = 5 NEW_LINE
printClosest ( N , X ) NEW_LINE
def maxMatch ( A , B ) : NEW_LINE
Aindex = { } NEW_LINE
diff = { } NEW_LINE
for i in range ( len ( A ) ) : NEW_LINE INDENT Aindex [ A [ i ] ] = i NEW_LINE DEDENT
for i in range ( len ( B ) ) : NEW_LINE
if i - Aindex [ B [ i ] ] < 0 : NEW_LINE INDENT if len ( A ) + i - Aindex [ B [ i ] ] not in diff : NEW_LINE INDENT diff [ len ( A ) + i - Aindex [ B [ i ] ] ] = 1 NEW_LINE DEDENT else : NEW_LINE INDENT diff [ len ( A ) + i - Aindex [ B [ i ] ] ] += 1 NEW_LINE DEDENT DEDENT
else : NEW_LINE INDENT if i - Aindex [ B [ i ] ] not in diff : NEW_LINE INDENT diff [ i - Aindex [ B [ i ] ] ] = 1 NEW_LINE DEDENT else : NEW_LINE INDENT diff [ i - Aindex [ B [ i ] ] ] += 1 NEW_LINE DEDENT DEDENT
return max ( diff . values ( ) ) NEW_LINE
A = [ 5 , 3 , 7 , 9 , 8 ] NEW_LINE B = [ 8 , 7 , 3 , 5 , 9 ] NEW_LINE
print ( maxMatch ( A , B ) ) NEW_LINE
def isinRange ( board ) : NEW_LINE INDENT N = 9 NEW_LINE DEDENT
INDENT for i in range ( 0 , N ) : NEW_LINE INDENT for j in range ( 0 , N ) : NEW_LINE DEDENT DEDENT
def isValidSudoku ( board ) : NEW_LINE INDENT N = 9 NEW_LINE DEDENT
INDENT if ( isinRange ( board ) == False ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT
INDENT unique = [ False ] * ( N + 1 ) NEW_LINE DEDENT
INDENT for i in range ( 0 , N ) : NEW_LINE DEDENT
for m in range ( 0 , N + 1 ) : NEW_LINE unique [ m ] = False NEW_LINE
for j in range ( 0 , N ) : NEW_LINE
Z = board [ i ] [ j ] NEW_LINE
if ( unique [ Z ] == True ) : NEW_LINE INDENT return False NEW_LINE DEDENT unique [ Z ] = True NEW_LINE
INDENT for i in range ( 0 , N ) : NEW_LINE DEDENT
for m in range ( 0 , N + 1 ) : NEW_LINE unique [ m ] = False NEW_LINE
for j in range ( 0 , N ) : NEW_LINE
Z = board [ j ] [ i ] NEW_LINE
if ( unique [ Z ] == True ) : NEW_LINE INDENT return False NEW_LINE DEDENT unique [ Z ] = True NEW_LINE
INDENT for i in range ( 0 , N - 2 , 3 ) : NEW_LINE DEDENT
for j in range ( 0 , N - 2 , 3 ) : NEW_LINE
for m in range ( 0 , N + 1 ) : NEW_LINE INDENT unique [ m ] = False NEW_LINE DEDENT
for k in range ( 0 , 3 ) : NEW_LINE INDENT for l in range ( 0 , 3 ) : NEW_LINE DEDENT
X = i + k NEW_LINE
Y = j + l NEW_LINE
Z = board [ X ] [ Y ] NEW_LINE
if ( unique [ Z ] == True ) : NEW_LINE INDENT return False NEW_LINE DEDENT unique [ Z ] = True NEW_LINE
INDENT return True NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT board = [ [ 7 , 9 , 2 , 1 , 5 , 4 , 3 , 8 , 6 ] , [ 6 , 4 , 3 , 8 , 2 , 7 , 1 , 5 , 9 ] , [ 8 , 5 , 1 , 3 , 9 , 6 , 7 , 2 , 4 ] , [ 2 , 6 , 5 , 9 , 7 , 3 , 8 , 4 , 1 ] , [ 4 , 8 , 9 , 5 , 6 , 1 , 2 , 7 , 3 ] , [ 3 , 1 , 7 , 4 , 8 , 2 , 9 , 6 , 5 ] , [ 1 , 3 , 6 , 7 , 4 , 8 , 5 , 9 , 2 ] , [ 9 , 7 , 4 , 2 , 1 , 5 , 6 , 3 , 8 ] , [ 5 , 2 , 8 , 6 , 3 , 9 , 4 , 1 , 7 ] ] NEW_LINE if ( isValidSudoku ( board ) ) : NEW_LINE INDENT print ( " Valid " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Not ▁ Valid " ) NEW_LINE DEDENT DEDENT
def palindrome ( a , i , j ) : NEW_LINE INDENT while ( i < j ) : NEW_LINE DEDENT
if ( a [ i ] != a [ j ] ) : NEW_LINE INDENT return False NEW_LINE DEDENT
i += 1 NEW_LINE j -= 1 NEW_LINE
return True NEW_LINE
def findSubArray ( arr , k ) : NEW_LINE INDENT n = len ( arr ) NEW_LINE DEDENT
for i in range ( n - k + 1 ) : NEW_LINE INDENT if ( palindrome ( arr , i , i + k - 1 ) ) : NEW_LINE INDENT return i NEW_LINE DEDENT DEDENT
return - 1 NEW_LINE
arr = [ 2 , 3 , 5 , 1 , 3 ] NEW_LINE k = 4 NEW_LINE ans = findSubArray ( arr , k ) NEW_LINE if ( ans == - 1 ) : NEW_LINE INDENT print ( - 1 ) NEW_LINE DEDENT else : NEW_LINE INDENT for i in range ( ans , ans + k ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT DEDENT
def isCrossed ( path ) : NEW_LINE INDENT if ( len ( path ) == 0 ) : NEW_LINE INDENT return bool ( False ) NEW_LINE DEDENT DEDENT
ans = bool ( False ) NEW_LINE
Set = set ( ) NEW_LINE
x , y = 0 , 0 NEW_LINE Set . add ( ( x , y ) ) NEW_LINE
for i in range ( len ( path ) ) : NEW_LINE
if ( path [ i ] == ' N ' ) : NEW_LINE INDENT Set . add ( ( x , y ) ) NEW_LINE y = y + 1 NEW_LINE DEDENT if ( path [ i ] == ' S ' ) : NEW_LINE INDENT Set . add ( ( x , y ) ) NEW_LINE y = y - 1 NEW_LINE DEDENT if ( path [ i ] == ' E ' ) : NEW_LINE INDENT Set . add ( ( x , y ) ) NEW_LINE x = x + 1 NEW_LINE DEDENT if ( path [ i ] == ' W ' ) : NEW_LINE INDENT Set . add ( ( x , y ) ) NEW_LINE x = x - 1 NEW_LINE DEDENT
if ( x , y ) in Set : NEW_LINE INDENT ans = bool ( True ) NEW_LINE break NEW_LINE DEDENT
if ( ans ) : NEW_LINE INDENT print ( " Crossed " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Not ▁ Crossed " ) NEW_LINE DEDENT
path = " NESW " NEW_LINE
isCrossed ( path ) NEW_LINE
from collections import deque NEW_LINE
def maxWidth ( N , M , cost , s ) : NEW_LINE
adj = [ [ ] for i in range ( N ) ] NEW_LINE for i in range ( M ) : NEW_LINE INDENT adj [ s [ i ] [ 0 ] ] . append ( s [ i ] [ 1 ] ) NEW_LINE DEDENT
result = 0 NEW_LINE
q = deque ( ) NEW_LINE
q . append ( 0 ) NEW_LINE
while ( len ( q ) > 0 ) : NEW_LINE
count = len ( q ) NEW_LINE
result = max ( count , result ) NEW_LINE
while ( count > 0 ) : NEW_LINE
temp = q . popleft ( ) NEW_LINE
for i in adj [ temp ] : NEW_LINE INDENT q . append ( i ) NEW_LINE DEDENT count -= 1 NEW_LINE
return result NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 11 NEW_LINE M = 10 NEW_LINE edges = [ ] NEW_LINE edges . append ( [ 0 , 1 ] ) NEW_LINE edges . append ( [ 0 , 2 ] ) NEW_LINE edges . append ( [ 0 , 3 ] ) NEW_LINE edges . append ( [ 1 , 4 ] ) NEW_LINE edges . append ( [ 1 , 5 ] ) NEW_LINE edges . append ( [ 3 , 6 ] ) NEW_LINE edges . append ( [ 4 , 7 ] ) NEW_LINE edges . append ( [ 6 , 1 ] ) NEW_LINE edges . append ( [ 6 , 8 ] ) NEW_LINE edges . append ( [ 6 , 9 ] ) NEW_LINE cost = [ 1 , 2 , - 1 , 3 , 4 , 5 , 8 , 2 , 6 , 12 , 7 ] NEW_LINE DEDENT
print ( maxWidth ( N , M , cost , edges ) ) NEW_LINE
MAX = 10000000 NEW_LINE
isPrime = [ True ] * ( MAX + 1 ) NEW_LINE
primes = [ ] NEW_LINE
def SieveOfEratosthenes ( ) : NEW_LINE INDENT global isPrime NEW_LINE p = 2 NEW_LINE while p * p <= MAX : NEW_LINE DEDENT
if ( isPrime [ p ] == True ) : NEW_LINE
for i in range ( p * p , MAX + 1 , p ) : NEW_LINE INDENT isPrime [ i ] = False NEW_LINE DEDENT p += 1 NEW_LINE
for p in range ( 2 , MAX + 1 ) : NEW_LINE INDENT if ( isPrime [ p ] ) : NEW_LINE INDENT primes . append ( p ) NEW_LINE DEDENT DEDENT
def prime_search ( primes , diff ) : NEW_LINE
low = 0 NEW_LINE high = len ( primes ) - 1 NEW_LINE while ( low <= high ) : NEW_LINE INDENT mid = ( low + high ) // 2 NEW_LINE DEDENT
if ( primes [ mid ] == diff ) : NEW_LINE
return primes [ mid ] NEW_LINE
elif ( primes [ mid ] < diff ) : NEW_LINE
low = mid + 1 NEW_LINE
else : NEW_LINE INDENT res = primes [ mid ] NEW_LINE DEDENT
high = mid - 1 NEW_LINE
return res NEW_LINE
def minCost ( arr , n ) : NEW_LINE
SieveOfEratosthenes ( ) NEW_LINE
res = 0 NEW_LINE
for i in range ( 1 , n ) : NEW_LINE
if ( arr [ i ] < arr [ i - 1 ] ) : NEW_LINE INDENT diff = arr [ i - 1 ] - arr [ i ] NEW_LINE DEDENT
closest_prime = prime_search ( primes , diff ) NEW_LINE
res += closest_prime NEW_LINE
arr [ i ] += closest_prime NEW_LINE
return res NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
arr = [ 2 , 1 , 5 , 4 , 3 ] NEW_LINE n = 5 NEW_LINE
print ( minCost ( arr , n ) ) NEW_LINE
def count ( s ) : NEW_LINE
cnt = 0 NEW_LINE
for c in s : NEW_LINE INDENT if c == '0' : NEW_LINE INDENT cnt += 1 NEW_LINE DEDENT DEDENT
if ( cnt % 3 != 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT res = 0 NEW_LINE k = cnt // 3 NEW_LINE sum = 0 NEW_LINE
mp = { } NEW_LINE
for i in range ( len ( s ) ) : NEW_LINE
if s [ i ] == '0' : NEW_LINE INDENT sum += 1 NEW_LINE DEDENT
if ( sum == 2 * k and k in mp and i < len ( s ) - 1 and i > 0 ) : NEW_LINE INDENT res += mp [ k ] NEW_LINE DEDENT
if sum in mp : NEW_LINE INDENT mp [ sum ] += 1 NEW_LINE DEDENT else : NEW_LINE INDENT mp [ sum ] = 1 NEW_LINE DEDENT
return res NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
st = "01010" NEW_LINE
print ( count ( st ) ) NEW_LINE
def splitstring ( s ) : NEW_LINE INDENT n = len ( s ) NEW_LINE DEDENT
zeros = 0 NEW_LINE for i in range ( n ) : NEW_LINE INDENT if s [ i ] == '0' : NEW_LINE INDENT zeros += 1 NEW_LINE DEDENT DEDENT
if zeros % 3 != 0 : NEW_LINE INDENT return 0 NEW_LINE DEDENT
if zeros == 0 : NEW_LINE INDENT return ( ( n - 1 ) * ( n - 2 ) ) // 2 NEW_LINE DEDENT
zerosInEachSubstring = zeros // 3 NEW_LINE
waysOfFirstCut , waysOfSecondCut = 0 , 0 NEW_LINE
count = 0 NEW_LINE
for i in range ( n ) : NEW_LINE
' NEW_LINE INDENT if s [ i ] == '0' : NEW_LINE INDENT count += 1 NEW_LINE DEDENT DEDENT
if ( count == zerosInEachSubstring ) : NEW_LINE INDENT waysOfFirstCut += 1 NEW_LINE DEDENT
elif ( count == 2 * zerosInEachSubstring ) : NEW_LINE INDENT waysOfSecondCut += 1 NEW_LINE DEDENT
return waysOfFirstCut * waysOfSecondCut NEW_LINE
s = "01010" NEW_LINE
print ( " The ▁ number ▁ of ▁ ways ▁ to ▁ split ▁ is " , splitstring ( s ) ) NEW_LINE
def canTransform ( str1 , str2 ) : NEW_LINE INDENT s1 = " " NEW_LINE s2 = " " NEW_LINE DEDENT
for c in str1 : NEW_LINE INDENT if ( c != ' C ' ) : NEW_LINE INDENT s1 += c NEW_LINE DEDENT DEDENT for c in str2 : NEW_LINE INDENT if ( c != ' C ' ) : NEW_LINE INDENT s2 += c NEW_LINE DEDENT DEDENT
if ( s1 != s2 ) : NEW_LINE INDENT return False NEW_LINE DEDENT i = 0 NEW_LINE j = 0 NEW_LINE n = len ( str1 ) NEW_LINE
while ( i < n and j < n ) : NEW_LINE INDENT if ( str1 [ i ] == ' C ' ) : NEW_LINE INDENT i += 1 NEW_LINE DEDENT elif ( str2 [ j ] == ' C ' ) : NEW_LINE INDENT j += 1 NEW_LINE DEDENT DEDENT
else : NEW_LINE INDENT if ( ( str1 [ i ] == ' A ' and i < j ) or ( str1 [ i ] == ' B ' and i > j ) ) : NEW_LINE INDENT return False NEW_LINE DEDENT i += 1 NEW_LINE j += 1 NEW_LINE DEDENT return True NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT str1 = " BCCABCBCA " NEW_LINE str2 = " CBACCBBAC " NEW_LINE DEDENT
if ( canTransform ( str1 , str2 ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
def maxsubstringLength ( S , N ) : NEW_LINE INDENT arr = [ 0 ] * N NEW_LINE DEDENT
for i in range ( N ) : NEW_LINE INDENT if ( S [ i ] == ' a ' or S [ i ] == ' e ' or S [ i ] == ' i ' or S [ i ] == ' o ' or S [ i ] == ' u ' ) : NEW_LINE INDENT arr [ i ] = 1 NEW_LINE DEDENT else : NEW_LINE INDENT arr [ i ] = - 1 NEW_LINE DEDENT DEDENT
maxLen = 0 NEW_LINE
curr_sum = 0 NEW_LINE
hash = { } NEW_LINE
for i in range ( N ) : NEW_LINE INDENT curr_sum += arr [ i ] NEW_LINE DEDENT
if ( curr_sum == 0 ) : NEW_LINE
maxLen = max ( maxLen , i + 1 ) NEW_LINE
if ( curr_sum in hash . keys ( ) ) : NEW_LINE INDENT maxLen = max ( maxLen , i - hash [ curr_sum ] ) NEW_LINE DEDENT
else : NEW_LINE INDENT hash [ curr_sum ] = i NEW_LINE DEDENT
return maxLen NEW_LINE
S = " geeksforgeeks " NEW_LINE n = len ( S ) NEW_LINE print ( maxsubstringLength ( S , n ) ) NEW_LINE
mat = [ [ 0 for x in range ( 1001 ) ] for y in range ( 1001 ) ] NEW_LINE
dx = [ 0 , - 1 , - 1 , - 1 , 0 , 1 , 1 , 1 ] NEW_LINE dy = [ 1 , 1 , 0 , - 1 , - 1 , - 1 , 0 , 1 ] NEW_LINE
def FindMinimumDistance ( ) : NEW_LINE INDENT global x , y , r , c NEW_LINE DEDENT
q = [ ] NEW_LINE
q . append ( [ x , y ] ) NEW_LINE mat [ x ] [ y ] = 0 NEW_LINE
while ( len ( q ) != 0 ) : NEW_LINE
x = q [ 0 ] [ 0 ] NEW_LINE y = q [ 0 ] [ 1 ] NEW_LINE
q . pop ( 0 ) NEW_LINE for i in range ( 8 ) : NEW_LINE INDENT a = x + dx [ i ] NEW_LINE b = y + dy [ i ] NEW_LINE DEDENT
if ( a < 0 or a >= r or b >= c or b < 0 ) : NEW_LINE INDENT continue NEW_LINE DEDENT
if ( mat [ a ] [ b ] == 0 ) : NEW_LINE
mat [ a ] [ b ] = mat [ x ] [ y ] + 1 NEW_LINE
q . append ( [ a , b ] ) NEW_LINE
r = 5 NEW_LINE c = 5 NEW_LINE x = 1 NEW_LINE y = 1 NEW_LINE t = x NEW_LINE l = y NEW_LINE mat [ x ] [ y ] = 0 NEW_LINE FindMinimumDistance ( ) NEW_LINE mat [ t ] [ l ] = 0 NEW_LINE
for i in range ( r ) : NEW_LINE INDENT for j in range ( c ) : NEW_LINE INDENT print ( mat [ i ] [ j ] , end = " ▁ " ) NEW_LINE DEDENT print ( ) NEW_LINE DEDENT
def minOperations ( S , K ) : NEW_LINE
ans = 0 NEW_LINE
for i in range ( K ) : NEW_LINE
zero , one = 0 , 0 NEW_LINE
for j in range ( i , len ( S ) , K ) : NEW_LINE
if ( S [ j ] == '0' ) : NEW_LINE INDENT zero += 1 NEW_LINE DEDENT
else : NEW_LINE INDENT one += 1 NEW_LINE DEDENT
ans += min ( zero , one ) NEW_LINE
return ans NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT s = "110100101" NEW_LINE K = 3 NEW_LINE print ( minOperations ( s , K ) ) NEW_LINE DEDENT
def missingElement ( arr , n ) : NEW_LINE
max_ele = arr [ 0 ] NEW_LINE
min_ele = arr [ 0 ] NEW_LINE
x = 0 NEW_LINE
d = 0 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT if ( arr [ i ] > max_ele ) : NEW_LINE INDENT max_ele = arr [ i ] NEW_LINE DEDENT if ( arr [ i ] < min_ele ) : NEW_LINE INDENT min_ele = arr [ i ] NEW_LINE DEDENT DEDENT
d = ( max_ele - min_ele ) // n NEW_LINE
for i in range ( n ) : NEW_LINE INDENT x = x ^ arr [ i ] NEW_LINE DEDENT
for i in range ( n + 1 ) : NEW_LINE INDENT x = x ^ ( min_ele + ( i * d ) ) NEW_LINE DEDENT
return x NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
arr = [ 12 , 3 , 6 , 15 , 18 ] NEW_LINE n = len ( arr ) NEW_LINE
element = missingElement ( arr , n ) NEW_LINE
print ( element ) NEW_LINE
def Printksubstring ( str1 , n , k ) : NEW_LINE
total = int ( ( n * ( n + 1 ) ) / 2 ) NEW_LINE
if ( k > total ) : NEW_LINE INDENT print ( " - 1" ) NEW_LINE return NEW_LINE DEDENT
substring = [ 0 for i in range ( n + 1 ) ] NEW_LINE substring [ 0 ] = 0 NEW_LINE
temp = n NEW_LINE for i in range ( 1 , n + 1 , 1 ) : NEW_LINE
substring [ i ] = substring [ i - 1 ] + temp NEW_LINE temp -= 1 NEW_LINE
l = 1 NEW_LINE h = n NEW_LINE start = 0 NEW_LINE while ( l <= h ) : NEW_LINE INDENT m = int ( ( l + h ) / 2 ) NEW_LINE if ( substring [ m ] > k ) : NEW_LINE INDENT start = m NEW_LINE h = m - 1 NEW_LINE DEDENT elif ( substring [ m ] < k ) : NEW_LINE INDENT l = m + 1 NEW_LINE DEDENT else : NEW_LINE INDENT start = m NEW_LINE break NEW_LINE DEDENT DEDENT
end = n - ( substring [ start ] - k ) NEW_LINE
for i in range ( start - 1 , end ) : NEW_LINE INDENT print ( str1 [ i ] , end = " " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT str1 = " abc " NEW_LINE k = 4 NEW_LINE n = len ( str1 ) NEW_LINE Printksubstring ( str1 , n , k ) NEW_LINE DEDENT
def LowerInsertionPoint ( arr , n , X ) : NEW_LINE
if ( X < arr [ 0 ] ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT elif ( X > arr [ n - 1 ] ) : NEW_LINE INDENT return n NEW_LINE DEDENT lowerPnt = 0 NEW_LINE i = 1 NEW_LINE while ( i < n and arr [ i ] < X ) : NEW_LINE INDENT lowerPnt = i NEW_LINE i = i * 2 NEW_LINE DEDENT
while ( lowerPnt < n and arr [ lowerPnt ] < X ) : NEW_LINE INDENT lowerPnt += 1 NEW_LINE DEDENT return lowerPnt NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 2 , 3 , 4 , 4 , 5 , 6 , 7 , 9 ] NEW_LINE n = len ( arr ) NEW_LINE X = 4 NEW_LINE print ( LowerInsertionPoint ( arr , n , X ) ) NEW_LINE DEDENT
def getCount ( M , N ) : NEW_LINE INDENT count = 0 ; NEW_LINE DEDENT
if ( M == 1 ) : NEW_LINE INDENT return N ; NEW_LINE DEDENT
if ( N == 1 ) : NEW_LINE INDENT return M ; NEW_LINE DEDENT if ( N > M ) : NEW_LINE
for i in range ( 1 , M + 1 ) : NEW_LINE INDENT numerator = N * i - N + M - i ; NEW_LINE denominator = M - 1 ; NEW_LINE DEDENT
if ( numerator % denominator == 0 ) : NEW_LINE INDENT j = numerator / denominator ; NEW_LINE DEDENT
if ( j >= 1 and j <= N ) : NEW_LINE INDENT count += 1 ; NEW_LINE DEDENT else : NEW_LINE
for j in range ( 1 , N + 1 ) : NEW_LINE INDENT numerator = M * j - M + N - j ; NEW_LINE denominator = N - 1 ; NEW_LINE DEDENT
if ( numerator % denominator == 0 ) : NEW_LINE INDENT i = numerator / denominator ; NEW_LINE DEDENT
if ( i >= 1 and i <= M ) : NEW_LINE INDENT count += 1 ; NEW_LINE DEDENT return count ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT M , N = 3 , 5 ; NEW_LINE print ( getCount ( M , N ) ) ; NEW_LINE DEDENT
import sys NEW_LINE
def swapElement ( arr1 , arr2 , n ) : NEW_LINE
wrongIdx = 0 ; NEW_LINE for i in range ( 1 , n ) : NEW_LINE INDENT if ( arr1 [ i ] < arr1 [ i - 1 ] ) : NEW_LINE INDENT wrongIdx = i NEW_LINE DEDENT DEDENT maximum = - ( sys . maxsize - 1 ) NEW_LINE maxIdx = - 1 NEW_LINE res = False NEW_LINE
for i in range ( n ) : NEW_LINE INDENT if ( arr2 [ i ] > maximum and arr2 [ i ] >= arr1 [ wrongIdx - 1 ] ) : NEW_LINE INDENT if ( wrongIdx + 1 <= n - 1 and arr2 [ i ] <= arr1 [ wrongIdx + 1 ] ) : NEW_LINE INDENT maximum = arr2 [ i ] NEW_LINE maxIdx = i NEW_LINE res = True NEW_LINE DEDENT DEDENT DEDENT
if ( res ) : NEW_LINE INDENT ( arr1 [ wrongIdx ] , arr2 [ maxIdx ] ) = ( arr2 [ maxIdx ] , arr1 [ wrongIdx ] ) NEW_LINE DEDENT return res NEW_LINE
def getSortedArray ( arr1 , arr2 , n ) : NEW_LINE INDENT if ( swapElement ( arr1 , arr2 , n ) ) : NEW_LINE INDENT for i in range ( n ) : NEW_LINE INDENT print ( arr1 [ i ] , end = " ▁ " ) NEW_LINE DEDENT DEDENT else : NEW_LINE INDENT print ( " Not ▁ Possible " ) NEW_LINE DEDENT DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr1 = [ 1 , 3 , 7 , 4 , 10 ] NEW_LINE arr2 = [ 2 , 1 , 6 , 8 , 9 ] NEW_LINE n = len ( arr1 ) NEW_LINE getSortedArray ( arr1 , arr2 , n ) NEW_LINE DEDENT
def middleOfThree ( a , b , c ) : NEW_LINE
if a > b : NEW_LINE INDENT if ( b > c ) : NEW_LINE INDENT return b NEW_LINE DEDENT elif ( a > c ) : NEW_LINE INDENT return c NEW_LINE DEDENT else : NEW_LINE INDENT return a NEW_LINE DEDENT DEDENT else : NEW_LINE
if ( a > c ) : NEW_LINE INDENT return a NEW_LINE DEDENT elif ( b > c ) : NEW_LINE INDENT return c NEW_LINE DEDENT else : NEW_LINE INDENT return b NEW_LINE DEDENT
a = 20 NEW_LINE b = 30 NEW_LINE c = 40 NEW_LINE print ( middleOfThree ( a , b , c ) ) NEW_LINE
def transpose ( mat , row , col ) : NEW_LINE
tr = [ [ 0 for i in range ( row ) ] for i in range ( col ) ] NEW_LINE
for i in range ( row ) : NEW_LINE
for j in range ( col ) : NEW_LINE
tr [ j ] [ i ] = mat [ i ] [ j ] NEW_LINE return tr NEW_LINE
def RowWiseSort ( B ) : NEW_LINE
for i in range ( len ( B ) ) : NEW_LINE
B [ i ] = sorted ( B [ i ] ) NEW_LINE return B NEW_LINE
def sortCol ( mat , N , M ) : NEW_LINE
B = transpose ( mat , N , M ) NEW_LINE
B = RowWiseSort ( B ) NEW_LINE
mat = transpose ( B , M , N ) NEW_LINE
for i in range ( N ) : NEW_LINE INDENT for j in range ( M ) : NEW_LINE INDENT print ( mat [ i ] [ j ] , end = " ▁ " ) NEW_LINE DEDENT print ( ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE
mat = [ [ 1 , 6 , 10 ] , [ 8 , 5 , 9 ] , [ 9 , 4 , 15 ] , [ 7 , 3 , 60 ] ] NEW_LINE N = len ( mat ) NEW_LINE M = len ( mat [ 0 ] ) NEW_LINE
sortCol ( mat , N , M ) NEW_LINE
def largestArea ( N , M , H , V , h , v ) : NEW_LINE
INDENT s1 = set ( [ ] ) ; NEW_LINE s2 = set ( [ ] ) ; NEW_LINE DEDENT
INDENT for i in range ( 1 , N + 2 ) : NEW_LINE INDENT s1 . add ( i ) ; NEW_LINE DEDENT DEDENT
INDENT for i in range ( 1 , M + 2 ) : NEW_LINE INDENT s2 . add ( i ) ; NEW_LINE DEDENT DEDENT
INDENT for i in range ( h ) : NEW_LINE INDENT s1 . remove ( H [ i ] ) ; NEW_LINE DEDENT DEDENT
INDENT for i in range ( v ) : NEW_LINE INDENT s2 . remove ( V [ i ] ) ; NEW_LINE DEDENT DEDENT
INDENT list1 = [ 0 ] * len ( s1 ) NEW_LINE list2 = [ 0 ] * len ( s2 ) ; NEW_LINE i = 0 ; NEW_LINE for it1 in s1 : NEW_LINE INDENT list1 [ i ] = it1 ; NEW_LINE i += 1 NEW_LINE DEDENT i = 0 ; NEW_LINE for it2 in s2 : NEW_LINE INDENT list2 [ i ] = it2 NEW_LINE i += 1 NEW_LINE DEDENT DEDENT
INDENT list1 . sort ( ) ; NEW_LINE list2 . sort ( ) ; NEW_LINE maxH = 0 NEW_LINE p1 = 0 NEW_LINE maxV = 0 NEW_LINE p2 = 0 ; NEW_LINE DEDENT
INDENT for j in range ( len ( s1 ) ) : NEW_LINE INDENT maxH = max ( maxH , list1 [ j ] - p1 ) ; NEW_LINE p1 = list1 [ j ] ; NEW_LINE DEDENT DEDENT
INDENT for j in range ( len ( s2 ) ) : NEW_LINE INDENT maxV = max ( maxV , list2 [ j ] - p2 ) ; NEW_LINE p2 = list2 [ j ] ; NEW_LINE DEDENT DEDENT
INDENT print ( ( maxV * maxH ) ) NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE
INDENT N = 3 NEW_LINE M = 3 ; NEW_LINE DEDENT
INDENT H = [ 2 ] NEW_LINE V = [ 2 ] ; NEW_LINE h = len ( H ) NEW_LINE v = len ( V ) ; NEW_LINE DEDENT
INDENT largestArea ( N , M , H , V , h , v ) ; NEW_LINE DEDENT
def checkifSorted ( A , B , N ) : NEW_LINE
INDENT flag = False NEW_LINE DEDENT
INDENT for i in range ( N - 1 ) : NEW_LINE DEDENT
if ( A [ i ] > A [ i + 1 ] ) : NEW_LINE
flag = True NEW_LINE break NEW_LINE
INDENT if ( not flag ) : NEW_LINE INDENT return True NEW_LINE DEDENT DEDENT
INDENT count = 0 NEW_LINE DEDENT
INDENT for i in range ( N ) : NEW_LINE DEDENT
if ( B [ i ] == 0 ) : NEW_LINE
count += 1 NEW_LINE break NEW_LINE
INDENT for i in range ( N ) : NEW_LINE DEDENT
if B [ i ] : NEW_LINE count += 1 NEW_LINE break NEW_LINE
INDENT if ( count == 2 ) : NEW_LINE INDENT return True NEW_LINE DEDENT return False NEW_LINE DEDENT
A = [ 3 , 1 , 2 ] NEW_LINE
B = [ 0 , 1 , 1 ] NEW_LINE N = len ( A ) NEW_LINE
check = checkifSorted ( A , B , N ) NEW_LINE
if ( check ) : NEW_LINE INDENT print ( " YES " ) NEW_LINE DEDENT
else : NEW_LINE INDENT print ( " NO " ) NEW_LINE DEDENT
def minSteps ( A , B , M , N ) : NEW_LINE INDENT if ( A [ 0 ] > B [ 0 ] ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT if ( B [ 0 ] > A [ 0 ] ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT DEDENT
if ( M <= N and A [ 0 ] == B [ 0 ] and A . count ( A [ 0 ] ) == M and B . count ( B [ 0 ] ) == N ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
for i in range ( 1 , N ) : NEW_LINE INDENT if ( B [ i ] > B [ 0 ] ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT DEDENT
for i in range ( 1 , M ) : NEW_LINE INDENT if ( A [ i ] < A [ 0 ] ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT DEDENT
for i in range ( 1 , M ) : NEW_LINE INDENT if ( A [ i ] > A [ 0 ] ) : NEW_LINE INDENT A [ 0 ] , B [ i ] = B [ i ] , A [ 0 ] NEW_LINE A [ 0 ] , B [ 0 ] = B [ 0 ] , A [ 0 ] NEW_LINE return 2 NEW_LINE DEDENT DEDENT
for i in range ( 1 , N ) : NEW_LINE INDENT if ( B [ i ] < B [ 0 ] ) : NEW_LINE INDENT A [ 0 ] , B [ i ] = B [ i ] , A [ 0 ] NEW_LINE A [ 0 ] , B [ 0 ] = B [ 0 ] , A [ 0 ] NEW_LINE return 2 NEW_LINE DEDENT DEDENT
return 0 NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT A = " adsfd " NEW_LINE B = " dffff " NEW_LINE M = len ( A ) NEW_LINE N = len ( B ) NEW_LINE print ( minSteps ( A , B , M , N ) ) NEW_LINE DEDENT
maxN = 201 ; NEW_LINE
n1 , n2 , n3 = 0 , 0 , 0 ; NEW_LINE
dp = [ [ [ 0 for i in range ( maxN ) ] for j in range ( maxN ) ] for j in range ( maxN ) ] ; NEW_LINE
def getMaxSum ( i , j , k , arr1 , arr2 , arr3 ) : NEW_LINE
cnt = 0 ; NEW_LINE if ( i >= n1 ) : NEW_LINE INDENT cnt += 1 ; NEW_LINE DEDENT if ( j >= n2 ) : NEW_LINE INDENT cnt += 1 ; NEW_LINE DEDENT if ( k >= n3 ) : NEW_LINE INDENT cnt += 1 ; NEW_LINE DEDENT
if ( cnt >= 2 ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT
if ( dp [ i ] [ j ] [ k ] != - 1 ) : NEW_LINE INDENT return dp [ i ] [ j ] [ k ] ; NEW_LINE DEDENT ans = 0 ; NEW_LINE
if ( i < n1 and j < n2 ) : NEW_LINE
ans = max ( ans , getMaxSum ( i + 1 , j + 1 , k , arr1 , arr2 , arr3 ) + arr1 [ i ] * arr2 [ j ] ) ; NEW_LINE if ( i < n1 and k < n3 ) : NEW_LINE ans = max ( ans , getMaxSum ( i + 1 , j , k + 1 , arr1 , arr2 , arr3 ) + arr1 [ i ] * arr3 [ k ] ) ; NEW_LINE if ( j < n2 and k < n3 ) : NEW_LINE ans = max ( ans , getMaxSum ( i , j + 1 , k + 1 , arr1 , arr2 , arr3 ) + arr2 [ j ] * arr3 [ k ] ) ; NEW_LINE
dp [ i ] [ j ] [ k ] = ans ; NEW_LINE
return dp [ i ] [ j ] [ k ] ; NEW_LINE def reverse ( tmp ) : NEW_LINE i , k , t = 0 , 0 , 0 ; NEW_LINE n = len ( tmp ) ; NEW_LINE for i in range ( n // 2 ) : NEW_LINE INDENT t = tmp [ i ] ; NEW_LINE tmp [ i ] = tmp [ n - i - 1 ] ; NEW_LINE tmp [ n - i - 1 ] = t ; NEW_LINE DEDENT
def maxProductSum ( arr1 , arr2 , arr3 ) : NEW_LINE
for i in range ( len ( dp ) ) : NEW_LINE INDENT for j in range ( len ( dp [ 0 ] ) ) : NEW_LINE INDENT for k in range ( len ( dp [ j ] [ 0 ] ) ) : NEW_LINE INDENT dp [ i ] [ j ] [ k ] = - 1 ; NEW_LINE DEDENT DEDENT DEDENT
arr1 . sort ( ) ; NEW_LINE reverse ( arr1 ) ; NEW_LINE arr2 . sort ( ) ; NEW_LINE reverse ( arr2 ) ; NEW_LINE arr3 . sort ( ) ; NEW_LINE reverse ( arr3 ) ; NEW_LINE return getMaxSum ( 0 , 0 , 0 , arr1 , arr2 , arr3 ) ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n1 = 2 ; NEW_LINE arr1 = [ 3 , 5 ] ; NEW_LINE n2 = 2 ; NEW_LINE arr2 = [ 2 , 1 ] ; NEW_LINE n3 = 3 ; NEW_LINE arr3 = [ 4 , 3 , 5 ] ; NEW_LINE print ( maxProductSum ( arr1 , arr2 , arr3 ) ) ; NEW_LINE DEDENT
def findTriplet ( arr , N ) : NEW_LINE
arr . sort ( ) NEW_LINE
i = N - 1 NEW_LINE while i - 2 >= 0 : NEW_LINE
if ( arr [ i - 2 ] + arr [ i - 1 ] > arr [ i ] ) : NEW_LINE INDENT flag = 1 NEW_LINE break NEW_LINE DEDENT i -= 1 NEW_LINE
if ( flag ) : NEW_LINE
print ( arr [ i - 2 ] , arr [ i - 1 ] , arr [ i ] ) NEW_LINE
else : NEW_LINE INDENT print ( - 1 ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 4 , 2 , 10 , 3 , 5 ] NEW_LINE N = len ( arr ) NEW_LINE findTriplet ( arr , N ) NEW_LINE DEDENT
def numberofpairs ( arr , N ) : NEW_LINE
answer = 0 NEW_LINE
arr . sort ( ) NEW_LINE
minDiff = 10000000 NEW_LINE for i in range ( 0 , N - 1 ) : NEW_LINE
minDiff = min ( minDiff , arr [ i + 1 ] - arr [ i ] ) NEW_LINE for i in range ( 0 , N - 1 ) : NEW_LINE if arr [ i + 1 ] - arr [ i ] == minDiff : NEW_LINE
answer += 1 NEW_LINE
return answer NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
arr = [ 4 , 2 , 1 , 3 ] NEW_LINE N = len ( arr ) NEW_LINE
print ( numberofpairs ( arr , N ) ) NEW_LINE
max_length = 0 NEW_LINE
store = [ ] NEW_LINE
ans = [ ] NEW_LINE
def find_max_length ( arr , index , sum , k ) : NEW_LINE INDENT global max_length NEW_LINE sum = sum + arr [ index ] NEW_LINE store . append ( arr [ index ] ) NEW_LINE if ( sum == k ) : NEW_LINE INDENT if ( max_length < len ( store ) ) : NEW_LINE DEDENT DEDENT
max_length = len ( store ) NEW_LINE
ans = store NEW_LINE for i in range ( index + 1 , len ( arr ) ) : NEW_LINE if ( sum + arr [ i ] <= k ) : NEW_LINE
find_max_length ( arr , i , sum , k ) NEW_LINE
store . pop ( ) NEW_LINE
else : NEW_LINE INDENT return NEW_LINE DEDENT return NEW_LINE def longestSubsequence ( arr , n , k ) : NEW_LINE
arr . sort ( ) NEW_LINE
for i in range ( n ) : NEW_LINE
if ( max_length >= n - i ) : NEW_LINE INDENT break NEW_LINE DEDENT store . clear ( ) NEW_LINE find_max_length ( arr , i , 0 , k ) NEW_LINE return max_length NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ - 3 , 0 , 1 , 1 , 2 ] NEW_LINE n = len ( arr ) NEW_LINE k = 1 NEW_LINE print ( longestSubsequence ( arr , n , k ) ) NEW_LINE DEDENT
def sortArray ( A , N ) : NEW_LINE
if ( N % 4 == 0 or N % 4 == 1 ) : NEW_LINE
for i in range ( N // 2 ) : NEW_LINE INDENT x = i NEW_LINE if ( i % 2 == 0 ) : NEW_LINE INDENT y = N - i - 2 NEW_LINE z = N - i - 1 NEW_LINE DEDENT DEDENT
A [ z ] = A [ y ] NEW_LINE A [ y ] = A [ x ] NEW_LINE A [ x ] = x + 1 NEW_LINE
print ( " Sorted ▁ Array : ▁ " , end = " " ) NEW_LINE for i in range ( N ) : NEW_LINE INDENT print ( A [ i ] , end = " ▁ " ) NEW_LINE DEDENT
else : NEW_LINE INDENT print ( " - 1" ) NEW_LINE DEDENT
A = [ 5 , 4 , 3 , 2 , 1 ] NEW_LINE N = len ( A ) NEW_LINE sortArray ( A , N ) NEW_LINE
def findK ( arr , size , N ) : NEW_LINE
arr = sorted ( arr ) NEW_LINE temp_sum = 0 NEW_LINE
for i in range ( size ) : NEW_LINE INDENT temp_sum += arr [ i ] NEW_LINE DEDENT
if ( N - temp_sum == arr [ i ] * ( size - i - 1 ) ) : NEW_LINE INDENT return arr [ i ] NEW_LINE DEDENT return - 1 NEW_LINE
arr = [ 3 , 1 , 10 , 4 , 8 ] NEW_LINE size = len ( arr ) NEW_LINE N = 16 NEW_LINE print ( findK ( arr , size , N ) ) NEW_LINE
def existsTriplet ( a , b , c , x , l1 , l2 , l3 ) : NEW_LINE
if ( l2 <= l1 and l2 <= l3 ) : NEW_LINE INDENT l1 , l2 = l2 , l1 NEW_LINE a , b = b , a NEW_LINE DEDENT elif ( l3 <= l1 and l3 <= l2 ) : NEW_LINE INDENT l1 , l3 = l3 , l1 NEW_LINE a , c = c , a NEW_LINE DEDENT
for i in range ( l1 ) : NEW_LINE
j = 0 NEW_LINE k = l3 - 1 NEW_LINE while ( j < l2 and k >= 0 ) : NEW_LINE
if ( a [ i ] + b [ j ] + c [ k ] == x ) : NEW_LINE INDENT return True NEW_LINE DEDENT if ( a [ i ] + b [ j ] + c [ k ] < x ) : NEW_LINE INDENT j += 1 NEW_LINE DEDENT else : NEW_LINE INDENT k -= 1 NEW_LINE DEDENT return False NEW_LINE
a = [ 2 , 7 , 8 , 10 , 15 ] NEW_LINE b = [ 1 , 6 , 7 , 8 ] NEW_LINE c = [ 4 , 5 , 5 ] NEW_LINE l1 = len ( a ) NEW_LINE l2 = len ( b ) NEW_LINE l3 = len ( c ) NEW_LINE x = 14 NEW_LINE if ( existsTriplet ( a , b , c , x , l1 , l2 , l3 ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
def printArr ( arr , n ) : NEW_LINE INDENT for i in range ( 0 , n ) : NEW_LINE INDENT print ( arr [ i ] , end = " " ) NEW_LINE DEDENT DEDENT
def compare ( num1 , num2 ) : NEW_LINE
A = str ( num1 ) NEW_LINE
B = str ( num2 ) NEW_LINE
return int ( A + B ) <= int ( B + A ) NEW_LINE
def printSmallest ( N , arr ) : NEW_LINE
sort ( arr ) NEW_LINE
printArr ( arr , N ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 5 , 6 , 2 , 9 , 21 , 1 ] NEW_LINE N = len ( arr ) NEW_LINE printSmallest ( N , arr ) NEW_LINE DEDENT
def stableSelectionSort ( a , n ) : NEW_LINE
for i in range ( n ) : NEW_LINE
min_idx = i NEW_LINE for j in range ( i + 1 , n ) : NEW_LINE INDENT if a [ min_idx ] > a [ j ] : NEW_LINE INDENT min_idx = j NEW_LINE DEDENT DEDENT
key = a [ min_idx ] NEW_LINE while min_idx > i : NEW_LINE INDENT a [ min_idx ] = a [ min_idx - 1 ] NEW_LINE min_idx -= 1 NEW_LINE DEDENT a [ i ] = key NEW_LINE def printArray ( a , n ) : NEW_LINE for i in range ( n ) : NEW_LINE print ( " % d " % a [ i ] , end = " ▁ " ) NEW_LINE
a = [ 4 , 5 , 3 , 2 , 4 , 1 ] NEW_LINE n = len ( a ) NEW_LINE stableSelectionSort ( a , n ) NEW_LINE printArray ( a , n ) NEW_LINE
def isPossible ( a , b , n , k ) : NEW_LINE
a . sort ( reverse = True ) NEW_LINE
b . sort ( ) NEW_LINE
for i in range ( n ) : NEW_LINE INDENT if ( a [ i ] + b [ i ] < k ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT return True NEW_LINE
a = [ 2 , 1 , 3 ] NEW_LINE b = [ 7 , 8 , 9 ] NEW_LINE k = 10 NEW_LINE n = len ( a ) NEW_LINE if ( isPossible ( a , b , n , k ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
def setBitCount ( num ) : NEW_LINE INDENT count = 0 NEW_LINE while ( num ) : NEW_LINE INDENT if ( num & 1 ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT num = num >> 1 NEW_LINE DEDENT return count NEW_LINE DEDENT
def sortBySetBitCount ( arr , n ) : NEW_LINE INDENT count = [ ] NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT count . append ( [ ( - 1 ) * setBitCount ( arr [ i ] ) , arr [ i ] ] ) NEW_LINE DEDENT count . sort ( key = lambda x : x [ 0 ] ) NEW_LINE for i in range ( len ( count ) ) : NEW_LINE INDENT print ( count [ i ] [ 1 ] , end = " ▁ " ) NEW_LINE DEDENT
arr = [ 1 , 2 , 3 , 4 , 5 , 6 ] NEW_LINE n = len ( arr ) NEW_LINE sortBySetBitCount ( arr , n ) NEW_LINE
def canReach ( s , L , R ) : NEW_LINE
dp = [ 0 for _ in range ( len ( s ) ) ] NEW_LINE
dp [ 0 ] = 1 NEW_LINE
pre = 0 NEW_LINE
for i in range ( 1 , len ( s ) ) : NEW_LINE
if ( i >= L ) : NEW_LINE INDENT pre += dp [ i - L ] NEW_LINE DEDENT
if ( i > R ) : NEW_LINE INDENT pre -= dp [ i - R - 1 ] NEW_LINE DEDENT dp [ i ] = ( pre > 0 ) and ( s [ i ] == '0' ) NEW_LINE
return dp [ len ( s ) - 1 ] NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT S = "01101110" NEW_LINE L = 2 NEW_LINE R = 3 NEW_LINE if canReach ( S , L , R ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT DEDENT
def maxXORUtil ( arr , N , xrr , orr ) : NEW_LINE
if ( N == 0 ) : NEW_LINE INDENT return xrr ^ orr NEW_LINE DEDENT
x = maxXORUtil ( arr , N - 1 , xrr ^ orr , arr [ N - 1 ] ) NEW_LINE
y = maxXORUtil ( arr , N - 1 , xrr , orr arr [ N - 1 ] ) NEW_LINE
return max ( x , y ) NEW_LINE
def maximumXOR ( arr , N ) : NEW_LINE
return maxXORUtil ( arr , N , 0 , 0 ) NEW_LINE
arr = 1 , 5 , 7 NEW_LINE N = len ( arr ) NEW_LINE print ( maximumXOR ( arr , N ) ) NEW_LINE
N = 10 ** 5 + 5 NEW_LINE
visited = [ 0 ] * N NEW_LINE
def construct_tree ( weights , n ) : NEW_LINE INDENT minimum = min ( weights ) NEW_LINE maximum = max ( weights ) NEW_LINE DEDENT
if ( minimum == maximum ) : NEW_LINE
print ( " No " ) NEW_LINE return NEW_LINE
else : NEW_LINE
print ( " Yes " ) NEW_LINE
root = weights [ 0 ] NEW_LINE
visited [ 1 ] = 1 NEW_LINE
for i in range ( n ) : NEW_LINE
if ( weights [ i ] != root and visited [ i + 1 ] == 0 ) : NEW_LINE INDENT print ( 1 , i + 1 ) NEW_LINE DEDENT
visited [ i + 1 ] = 1 NEW_LINE
notroot = 0 NEW_LINE for i in range ( n ) : NEW_LINE INDENT if ( weights [ i ] != root ) : NEW_LINE INDENT notroot = i + 1 NEW_LINE break NEW_LINE DEDENT DEDENT
for i in range ( n ) : NEW_LINE
if ( weights [ i ] == root and visited [ i + 1 ] == 0 ) : NEW_LINE INDENT print ( notroot , i + 1 ) NEW_LINE visited [ i + 1 ] = 1 NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT weights = [ 1 , 2 , 1 , 2 , 5 ] NEW_LINE N = len ( weights ) NEW_LINE DEDENT
construct_tree ( weights , N ) NEW_LINE
import sys NEW_LINE
def minCost ( s , k ) : NEW_LINE
n = len ( s ) NEW_LINE
ans = 0 NEW_LINE
for i in range ( k ) : NEW_LINE
a = [ 0 ] * 26 NEW_LINE for j in range ( i , n , k ) : NEW_LINE INDENT a [ ord ( s [ j ] ) - ord ( ' a ' ) ] += 1 NEW_LINE DEDENT
min_cost = sys . maxsize - 1 NEW_LINE
for ch in range ( 26 ) : NEW_LINE INDENT cost = 0 NEW_LINE DEDENT
for tr in range ( 26 ) : NEW_LINE INDENT cost += abs ( ch - tr ) * a [ tr ] NEW_LINE DEDENT
min_cost = min ( min_cost , cost ) NEW_LINE
ans += min_cost NEW_LINE
print ( ans ) NEW_LINE
S = " abcdefabc " NEW_LINE K = 3 NEW_LINE
minCost ( S , K ) NEW_LINE
def minAbsDiff ( N ) : NEW_LINE INDENT if ( N % 4 == 0 or N % 4 == 3 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT return 1 NEW_LINE DEDENT
N = 6 NEW_LINE print ( minAbsDiff ( N ) ) NEW_LINE
N = 10000 NEW_LINE
adj = { } NEW_LINE used = [ 0 for i in range ( N ) ] NEW_LINE max_matching = 0 NEW_LINE
def AddEdge ( u , v ) : NEW_LINE INDENT if u not in adj : NEW_LINE INDENT adj [ u ] = [ ] NEW_LINE DEDENT if v not in adj : NEW_LINE INDENT adj [ v ] = [ ] NEW_LINE DEDENT DEDENT
adj [ u ] . append ( v ) NEW_LINE
adj [ v ] . append ( u ) NEW_LINE
def Matching_dfs ( u , p ) : NEW_LINE INDENT global max_matching NEW_LINE for i in range ( len ( adj [ u ] ) ) : NEW_LINE DEDENT
if ( adj [ u ] [ i ] != p ) : NEW_LINE INDENT Matching_dfs ( adj [ u ] [ i ] , u ) NEW_LINE DEDENT
if ( not used [ u ] and not used [ p ] and p != 0 ) : NEW_LINE
max_matching += 1 NEW_LINE used [ u ] = 1 NEW_LINE used [ p ] = 1 NEW_LINE
def maxMatching ( ) : NEW_LINE
Matching_dfs ( 1 , 0 ) NEW_LINE
print ( max_matching ) NEW_LINE
n = 5 NEW_LINE
AddEdge ( 1 , 2 ) NEW_LINE AddEdge ( 1 , 3 ) NEW_LINE AddEdge ( 3 , 4 ) NEW_LINE AddEdge ( 3 , 5 ) NEW_LINE
maxMatching ( ) NEW_LINE
import sys NEW_LINE
def getMinCost ( A , B , N ) : NEW_LINE INDENT mini = sys . maxsize NEW_LINE for i in range ( N ) : NEW_LINE INDENT mini = min ( mini , min ( A [ i ] , B [ i ] ) ) NEW_LINE DEDENT DEDENT
return mini * ( 2 * N - 1 ) NEW_LINE
N = 3 NEW_LINE A = [ 1 , 4 , 2 ] NEW_LINE B = [ 10 , 6 , 12 ] NEW_LINE print ( getMinCost ( A , B , N ) ) NEW_LINE
def printVector ( arr ) : NEW_LINE INDENT if ( len ( arr ) != 1 ) : NEW_LINE DEDENT
for i in range ( len ( arr ) ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT print ( ) NEW_LINE
def findWays ( arr , i , n ) : NEW_LINE
if ( n == 0 ) : NEW_LINE INDENT printVector ( arr ) NEW_LINE DEDENT
for j in range ( i , n + 1 ) : NEW_LINE
arr . append ( j ) NEW_LINE
findWays ( arr , j , n - j ) NEW_LINE
del arr [ - 1 ] NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
n = 4 NEW_LINE
arr = [ ] NEW_LINE
findWays ( arr , 1 , n ) NEW_LINE
def Maximum_subsequence ( A , N ) : NEW_LINE
frequency = dict ( ) ; NEW_LINE
max_freq = 0 ; NEW_LINE for i in range ( N ) : NEW_LINE
if ( frequency [ it ] > max_freq ) : NEW_LINE INDENT max_freq = frequency [ it ] ; NEW_LINE DEDENT
print ( max_freq ) ; NEW_LINE
arr = [ 5 , 2 , 6 , 5 , 2 , 4 , 5 , 2 ] ; NEW_LINE Maximum_subsequence ( arr , len ( arr ) ) ; NEW_LINE
def DivideString ( s , n , k ) : NEW_LINE INDENT c = 0 NEW_LINE no = 1 NEW_LINE c1 = 0 NEW_LINE c2 = 0 NEW_LINE DEDENT
fr = [ 0 ] * 26 NEW_LINE ans = [ ] NEW_LINE for i in range ( n ) : NEW_LINE INDENT fr [ ord ( s [ i ] ) - ord ( ' a ' ) ] += 1 NEW_LINE DEDENT for i in range ( 26 ) : NEW_LINE
if ( fr [ i ] == k ) : NEW_LINE INDENT c += 1 NEW_LINE DEDENT
if ( fr [ i ] > k and fr [ i ] != 2 * k ) : NEW_LINE INDENT c1 += 1 NEW_LINE ch = chr ( ord ( ' a ' ) + i ) NEW_LINE DEDENT if ( fr [ i ] == 2 * k ) : NEW_LINE INDENT c2 += 1 NEW_LINE ch1 = chr ( ord ( ' a ' ) + i ) NEW_LINE DEDENT for i in range ( n ) : NEW_LINE ans . append ( "1" ) NEW_LINE mp = { } NEW_LINE if ( c % 2 == 0 or c1 > 0 or c2 > 0 ) : NEW_LINE for i in range ( n ) : NEW_LINE
if ( fr [ ord ( s [ i ] ) - ord ( ' a ' ) ] == k ) : NEW_LINE INDENT if ( s [ i ] in mp ) : NEW_LINE INDENT ans [ i ] = '2' NEW_LINE DEDENT else : NEW_LINE INDENT if ( no <= ( c // 2 ) ) : NEW_LINE INDENT ans [ i ] = '2' NEW_LINE no += 1 NEW_LINE mp [ s [ i ] ] = 1 NEW_LINE DEDENT DEDENT DEDENT
if ( c % 2 == 1 and c1 > 0 ) : NEW_LINE INDENT no = 1 NEW_LINE for i in range ( n ) : NEW_LINE INDENT if ( s [ i ] == ch and no <= k ) : NEW_LINE INDENT ans [ i ] = '2' NEW_LINE no += 1 NEW_LINE DEDENT DEDENT DEDENT
if ( c % 2 == 1 and c1 == 0 ) : NEW_LINE INDENT no = 1 NEW_LINE flag = 0 NEW_LINE for i in range ( n ) : NEW_LINE INDENT if ( s [ i ] == ch1 and no <= k ) : NEW_LINE INDENT ans [ i ] = '2' NEW_LINE no += 1 NEW_LINE DEDENT if ( fr [ s [ i ] - ' a ' ] == k and flag == 0 and ans [ i ] == '1' ) : NEW_LINE INDENT ans [ i ] = '2' NEW_LINE flag = 1 NEW_LINE DEDENT DEDENT DEDENT print ( " " . join ( ans ) ) NEW_LINE else : NEW_LINE
print ( " NO " ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT S = " abbbccc " NEW_LINE N = len ( S ) NEW_LINE K = 1 NEW_LINE DivideString ( S , N , K ) NEW_LINE DEDENT
def check ( S , prices , type1 , n ) : NEW_LINE
for j in range ( 0 , n ) : NEW_LINE INDENT for k in range ( j + 1 , n ) : NEW_LINE DEDENT
if ( ( type1 [ j ] == 0 and type1 [ k ] == 1 ) or ( type1 [ j ] == 1 and type1 [ k ] == 0 ) ) : NEW_LINE INDENT if ( prices [ j ] + prices [ k ] <= S ) : NEW_LINE INDENT return " Yes " ; NEW_LINE DEDENT DEDENT return " No " ; NEW_LINE
prices = [ 3 , 8 , 6 , 5 ] ; NEW_LINE type1 = [ 0 , 1 , 1 , 0 ] ; NEW_LINE S = 10 ; NEW_LINE n = 4 ; NEW_LINE
print ( check ( S , prices , type1 , n ) ) ; NEW_LINE
def getLargestSum ( N ) : NEW_LINE
for i in range ( 1 , int ( N ** ( 1 / 2 ) ) + 1 ) : NEW_LINE INDENT for j in range ( i + 1 , int ( N ** ( 1 / 2 ) ) + 1 ) : NEW_LINE DEDENT
k = N // j ; NEW_LINE a = k * i ; NEW_LINE b = k * j ; NEW_LINE
if ( a <= N and b <= N and a * b % ( a + b ) == 0 ) : NEW_LINE
max_sum = max ( max_sum , a + b ) ; NEW_LINE
return max_sum ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT N = 25 ; NEW_LINE max_sum = getLargestSum ( N ) ; NEW_LINE print ( max_sum ) ; NEW_LINE DEDENT
def encryptString ( string , n ) : NEW_LINE INDENT i , cnt = 0 , 0 NEW_LINE encryptedStr = " " NEW_LINE while i < n : NEW_LINE DEDENT
cnt = i + 1 NEW_LINE
while cnt > 0 : NEW_LINE INDENT encryptedStr += string [ i ] NEW_LINE cnt -= 1 NEW_LINE DEDENT i += 1 NEW_LINE return encryptedStr NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT string = " geeks " NEW_LINE n = len ( string ) NEW_LINE print ( encryptString ( string , n ) ) NEW_LINE DEDENT
def minDiff ( n , x , A ) : NEW_LINE INDENT mn = A [ 0 ] NEW_LINE mx = A [ 0 ] NEW_LINE DEDENT
for i in range ( 0 , n ) : NEW_LINE INDENT mn = min ( mn , A [ i ] ) NEW_LINE mx = max ( mx , A [ i ] ) NEW_LINE DEDENT
return max ( 0 , mx - mn - 2 * x ) NEW_LINE
n = 3 NEW_LINE x = 3 NEW_LINE A = [ 1 , 3 , 6 ] NEW_LINE
print ( minDiff ( n , x , A ) ) NEW_LINE
def swapCount ( s ) : NEW_LINE INDENT chars = s NEW_LINE DEDENT
countLeft = 0 NEW_LINE countRight = 0 NEW_LINE
swap = 0 NEW_LINE imbalance = 0 ; NEW_LINE for i in range ( len ( chars ) ) : NEW_LINE INDENT if chars [ i ] == ' [ ' : NEW_LINE DEDENT
countLeft += 1 NEW_LINE if imbalance > 0 : NEW_LINE
swap += imbalance NEW_LINE
imbalance -= 1 NEW_LINE elif chars [ i ] == ' ] ' : NEW_LINE
countRight += 1 NEW_LINE
imbalance = ( countRight - countLeft ) NEW_LINE return swap NEW_LINE
s = " [ ] ] [ ] [ " ; NEW_LINE print ( swapCount ( s ) ) NEW_LINE s = " [ [ ] [ ] ] " ; NEW_LINE print ( swapCount ( s ) ) NEW_LINE
def longestSubSequence ( A , N ) : NEW_LINE
dp = [ 0 ] * N NEW_LINE for i in range ( N ) : NEW_LINE
dp [ i ] = 1 NEW_LINE for j in range ( i ) : NEW_LINE
if ( A [ j ] [ 0 ] < A [ i ] [ 0 ] and A [ j ] [ 1 ] > A [ i ] [ 1 ] ) : NEW_LINE INDENT dp [ i ] = max ( dp [ i ] , dp [ j ] + 1 ) NEW_LINE DEDENT
print ( dp [ N - 1 ] ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
A = [ [ 1 , 2 ] , [ 2 , 2 ] , [ 3 , 1 ] ] NEW_LINE N = len ( A ) NEW_LINE
longestSubSequence ( A , N ) NEW_LINE
def findWays ( N , dp ) : NEW_LINE
if ( N == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT
if ( dp [ N ] != - 1 ) : NEW_LINE INDENT return dp [ N ] NEW_LINE DEDENT cnt = 0 NEW_LINE
for i in range ( 1 , 7 ) : NEW_LINE INDENT if ( N - i >= 0 ) : NEW_LINE INDENT cnt = ( cnt + findWays ( N - i , dp ) ) NEW_LINE DEDENT DEDENT
dp [ N ] = cnt NEW_LINE return dp [ N ] NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
N = 4 NEW_LINE
dp = [ - 1 ] * ( N + 1 ) NEW_LINE
print ( findWays ( N , dp ) ) NEW_LINE
def findWays ( N ) : NEW_LINE
dp = [ 0 ] * ( N + 1 ) ; NEW_LINE dp [ 0 ] = 1 ; NEW_LINE
for i in range ( 1 , N + 1 ) : NEW_LINE INDENT dp [ i ] = 0 ; NEW_LINE DEDENT
for j in range ( 1 , 7 ) : NEW_LINE INDENT if ( i - j >= 0 ) : NEW_LINE INDENT dp [ i ] = dp [ i ] + dp [ i - j ] ; NEW_LINE DEDENT DEDENT
print ( dp [ N ] ) ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
N = 4 ; NEW_LINE
findWays ( N ) ; NEW_LINE
INF = 1e9 + 9 NEW_LINE
class TrieNode ( ) : NEW_LINE INDENT def __init__ ( self ) : NEW_LINE INDENT self . child = [ None ] * 26 NEW_LINE DEDENT DEDENT
def insert ( idx , s , root ) : NEW_LINE INDENT temp = root NEW_LINE for i in range ( idx , len ( s ) ) : NEW_LINE DEDENT
if temp . child [ ord ( s [ i ] ) - ord ( ' a ' ) ] == None : NEW_LINE
temp . child [ ord ( s [ i ] ) - ord ( ' a ' ) ] = TrieNode ( ) NEW_LINE temp = temp . child [ ord ( s [ i ] ) - ord ( ' a ' ) ] NEW_LINE
def minCuts ( S1 , S2 ) : NEW_LINE INDENT n1 = len ( S1 ) NEW_LINE n2 = len ( S2 ) NEW_LINE DEDENT
root = TrieNode ( ) NEW_LINE for i in range ( n2 ) : NEW_LINE
insert ( i , S2 , root ) NEW_LINE
dp = [ INF ] * ( n1 + 1 ) NEW_LINE
dp [ 0 ] = 0 NEW_LINE for i in range ( n1 ) : NEW_LINE
temp = root NEW_LINE for j in range ( i + 1 , n1 + 1 ) : NEW_LINE INDENT if temp . child [ ord ( S1 [ j - 1 ] ) - ord ( ' a ' ) ] == None : NEW_LINE DEDENT
break NEW_LINE
dp [ j ] = min ( dp [ j ] , dp [ i ] + 1 ) NEW_LINE
temp = temp . child [ ord ( S1 [ j - 1 ] ) - ord ( ' a ' ) ] NEW_LINE
if dp [ n1 ] >= INF : NEW_LINE INDENT return - 1 NEW_LINE DEDENT else : NEW_LINE INDENT return dp [ n1 ] NEW_LINE DEDENT
S1 = " abcdab " NEW_LINE S2 = " dabc " NEW_LINE print ( minCuts ( S1 , S2 ) ) NEW_LINE
def largestSquare ( matrix , R , C , q_i , q_j , K , Q ) : NEW_LINE INDENT countDP = [ [ 0 for x in range ( C ) ] for x in range ( R ) ] NEW_LINE DEDENT
countDP [ 0 ] [ 0 ] = matrix [ 0 ] [ 0 ] NEW_LINE for i in range ( 1 , R ) : NEW_LINE INDENT countDP [ i ] [ 0 ] = ( countDP [ i - 1 ] [ 0 ] + matrix [ i ] [ 0 ] ) NEW_LINE DEDENT for j in range ( 1 , C ) : NEW_LINE INDENT countDP [ 0 ] [ j ] = ( countDP [ 0 ] [ j - 1 ] + matrix [ 0 ] [ j ] ) NEW_LINE DEDENT for i in range ( 1 , R ) : NEW_LINE INDENT for j in range ( 1 , C ) : NEW_LINE INDENT countDP [ i ] [ j ] = ( matrix [ i ] [ j ] + countDP [ i - 1 ] [ j ] + countDP [ i ] [ j - 1 ] - countDP [ i - 1 ] [ j - 1 ] ) NEW_LINE DEDENT DEDENT
for q in range ( 0 , Q ) : NEW_LINE INDENT i = q_i [ q ] NEW_LINE j = q_j [ q ] NEW_LINE DEDENT
min_dist = min ( i , j , R - i - 1 , C - j - 1 ) NEW_LINE ans = - 1 NEW_LINE l = 0 NEW_LINE u = min_dist NEW_LINE while ( l <= u ) : NEW_LINE INDENT mid = int ( ( l + u ) / 2 ) NEW_LINE x1 = i - mid NEW_LINE x2 = i + mid NEW_LINE y1 = j - mid NEW_LINE y2 = j + mid NEW_LINE DEDENT
count = countDP [ x2 ] [ y2 ] NEW_LINE if ( x1 > 0 ) : NEW_LINE INDENT count -= countDP [ x1 - 1 ] [ y2 ] NEW_LINE DEDENT if ( y1 > 0 ) : NEW_LINE INDENT count -= countDP [ x2 ] [ y1 - 1 ] NEW_LINE DEDENT if ( x1 > 0 and y1 > 0 ) : NEW_LINE INDENT count += countDP [ x1 - 1 ] [ y1 - 1 ] NEW_LINE DEDENT
if ( count <= K ) : NEW_LINE INDENT ans = 2 * mid + 1 NEW_LINE l = mid + 1 NEW_LINE DEDENT else : NEW_LINE INDENT u = mid - 1 NEW_LINE DEDENT print ( ans ) NEW_LINE
matrix = [ [ 1 , 0 , 1 , 0 , 0 ] , [ 1 , 0 , 1 , 1 , 1 ] , [ 1 , 1 , 1 , 1 , 1 ] , [ 1 , 0 , 0 , 1 , 0 ] ] NEW_LINE K = 9 NEW_LINE Q = 1 NEW_LINE q_i = [ 1 ] NEW_LINE q_j = [ 2 ] NEW_LINE largestSquare ( matrix , 4 , 5 , q_i , q_j , K , Q ) NEW_LINE
