def Conversion ( centi ) : NEW_LINE INDENT pixels = ( 96 * centi ) / 2.54 NEW_LINE print ( round ( pixels , 2 ) ) NEW_LINE DEDENT
centi = 15 NEW_LINE Conversion ( centi ) NEW_LINE
def xor_operations ( N , arr , M , K ) : NEW_LINE
if M < 0 or M >= N : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
if K < 0 or K >= N - M : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
for _ in range ( M ) : NEW_LINE
temp = [ ] NEW_LINE
for i in range ( len ( arr ) - 1 ) : NEW_LINE
value = arr [ i ] ^ arr [ i + 1 ] NEW_LINE
temp . append ( value ) NEW_LINE
arr = temp [ : ] NEW_LINE
ans = arr [ K ] NEW_LINE return ans NEW_LINE
N = 5 NEW_LINE
arr = [ 1 , 4 , 5 , 6 , 7 ] NEW_LINE M = 1 NEW_LINE K = 2 NEW_LINE
print ( xor_operations ( N , arr , M , K ) ) NEW_LINE
def canBreakN ( n ) : NEW_LINE
for i in range ( 2 , n ) : NEW_LINE
m = i * ( i + 1 ) // 2 NEW_LINE
if ( m > n ) : NEW_LINE INDENT break NEW_LINE DEDENT k = n - m NEW_LINE
if ( k % i ) : NEW_LINE INDENT continue NEW_LINE DEDENT
print ( i ) NEW_LINE return NEW_LINE
print ( " - 1" ) NEW_LINE
N = 12 NEW_LINE
canBreakN ( N ) NEW_LINE
import math NEW_LINE
def findCoprimePair ( N ) : NEW_LINE
for x in range ( 2 , int ( math . sqrt ( N ) ) + 1 ) : NEW_LINE INDENT if ( N % x == 0 ) : NEW_LINE DEDENT
while ( N % x == 0 ) : NEW_LINE INDENT N //= x NEW_LINE DEDENT if ( N > 1 ) : NEW_LINE
print ( x , N ) NEW_LINE return ; NEW_LINE
print ( " - 1" ) NEW_LINE
N = 45 NEW_LINE findCoprimePair ( N ) NEW_LINE
N = 25 NEW_LINE findCoprimePair ( N ) NEW_LINE
import math NEW_LINE MAX = 10000 NEW_LINE
primes = [ ] NEW_LINE
def sieveSundaram ( ) : NEW_LINE
marked = [ False ] * ( ( MAX // 2 ) + 1 ) NEW_LINE
for i in range ( 1 , ( ( int ( math . sqrt ( MAX ) ) - 1 ) // 2 ) + 1 ) : NEW_LINE INDENT j = ( i * ( i + 1 ) ) << 1 NEW_LINE while j <= ( MAX // 2 ) : NEW_LINE INDENT marked [ j ] = True NEW_LINE j = j + 2 * i + 1 NEW_LINE DEDENT DEDENT
primes . append ( 2 ) NEW_LINE
for i in range ( 1 , ( MAX // 2 ) + 1 ) : NEW_LINE INDENT if marked [ i ] == False : NEW_LINE INDENT primes . append ( 2 * i + 1 ) NEW_LINE DEDENT DEDENT
def isWasteful ( n ) : NEW_LINE INDENT if ( n == 1 ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT
original_no = n NEW_LINE sumDigits = 0 NEW_LINE while ( original_no > 0 ) : NEW_LINE INDENT sumDigits += 1 NEW_LINE original_no = original_no // 10 NEW_LINE DEDENT pDigit , count_exp , p = 0 , 0 , 0 NEW_LINE
i = 0 NEW_LINE while ( primes [ i ] <= ( n // 2 ) ) : NEW_LINE
while ( n % primes [ i ] == 0 ) : NEW_LINE
p = primes [ i ] NEW_LINE n = n // p NEW_LINE
count_exp += 1 NEW_LINE
while ( p > 0 ) : NEW_LINE INDENT pDigit += 1 NEW_LINE p = p // 10 NEW_LINE DEDENT
while ( count_exp > 1 ) : NEW_LINE INDENT pDigit += 1 NEW_LINE count_exp = count_exp // 10 NEW_LINE DEDENT i += 1 NEW_LINE
if ( n != 1 ) : NEW_LINE INDENT while ( n > 0 ) : NEW_LINE INDENT pDigit += 1 NEW_LINE n = n // 10 NEW_LINE DEDENT DEDENT
return bool ( pDigit > sumDigits ) NEW_LINE
def Solve ( N ) : NEW_LINE
for i in range ( 1 , N ) : NEW_LINE INDENT if ( isWasteful ( i ) ) : NEW_LINE INDENT print ( i , end = " ▁ " ) NEW_LINE DEDENT DEDENT
sieveSundaram ( ) NEW_LINE N = 10 NEW_LINE
Solve ( N ) NEW_LINE
def printhexaRec ( n ) : NEW_LINE INDENT if ( n == 0 or n == 1 or \ n == 2 or n == 3 or \ n == 4 or n == 5 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT elif ( n == 6 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT else : NEW_LINE INDENT return ( printhexaRec ( n - 1 ) + printhexaRec ( n - 2 ) + printhexaRec ( n - 3 ) + printhexaRec ( n - 4 ) + printhexaRec ( n - 5 ) + printhexaRec ( n - 6 ) ) NEW_LINE DEDENT DEDENT def printhexa ( n ) : NEW_LINE INDENT print ( printhexaRec ( n ) ) NEW_LINE DEDENT
n = 11 NEW_LINE printhexa ( n ) NEW_LINE
def printhexa ( n ) : NEW_LINE INDENT if ( n < 0 ) : NEW_LINE INDENT return NEW_LINE DEDENT DEDENT
first = 0 NEW_LINE second = 0 NEW_LINE third = 0 NEW_LINE fourth = 0 NEW_LINE fifth = 0 NEW_LINE sixth = 1 NEW_LINE
curr = 0 NEW_LINE if ( n < 6 ) : NEW_LINE INDENT print ( first ) NEW_LINE DEDENT elif ( n == 6 ) : NEW_LINE INDENT print ( sixth ) NEW_LINE DEDENT else : NEW_LINE
for i in range ( 6 , n ) : NEW_LINE INDENT curr = first + second + third + fourth + fifth + sixth NEW_LINE first = second NEW_LINE second = third NEW_LINE third = fourth NEW_LINE fourth = fifth NEW_LINE fifth = sixth NEW_LINE sixth = curr NEW_LINE DEDENT print ( curr ) NEW_LINE
n = 11 NEW_LINE printhexa ( n ) NEW_LINE
def smallestNumber ( N ) : NEW_LINE INDENT print ( ( N % 9 + 1 ) * pow ( 10 , ( N // 9 ) ) - 1 ) NEW_LINE DEDENT
N = 10 NEW_LINE smallestNumber ( N ) NEW_LINE
def isComposite ( n ) : NEW_LINE
if ( n <= 3 ) : NEW_LINE INDENT return False NEW_LINE DEDENT
if ( n % 2 == 0 or n % 3 == 0 ) : NEW_LINE INDENT return True NEW_LINE DEDENT i = 5 NEW_LINE while ( i * i <= n ) : NEW_LINE INDENT if ( n % i == 0 or n % ( i + 2 ) == 0 ) : NEW_LINE INDENT return True NEW_LINE DEDENT i = i + 6 NEW_LINE DEDENT return False NEW_LINE
def Compositorial_list ( n ) : NEW_LINE INDENT l = 0 NEW_LINE for i in range ( 4 , 10 ** 6 ) : NEW_LINE INDENT if l < n : NEW_LINE INDENT if isComposite ( i ) : NEW_LINE INDENT compo . append ( i ) NEW_LINE l += 1 NEW_LINE DEDENT DEDENT DEDENT DEDENT
def calculateCompositorial ( n ) : NEW_LINE
result = 1 NEW_LINE for i in range ( n ) : NEW_LINE INDENT result = result * compo [ i ] NEW_LINE DEDENT return result NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 5 NEW_LINE DEDENT
compo = [ ] NEW_LINE Compositorial_list ( n ) NEW_LINE print ( calculateCompositorial ( n ) ) NEW_LINE
b = [ 0 for i in range ( 50 ) ] NEW_LINE
def PowerArray ( n , k ) : NEW_LINE
count = 0 NEW_LINE
while ( k ) : NEW_LINE INDENT if ( k % n == 0 ) : NEW_LINE INDENT k //= n NEW_LINE count += 1 NEW_LINE DEDENT DEDENT
elif ( k % n == 1 ) : NEW_LINE INDENT k -= 1 NEW_LINE b [ count ] += 1 NEW_LINE DEDENT
if ( b [ count ] > 1 ) : NEW_LINE INDENT print ( - 1 ) NEW_LINE return 0 NEW_LINE DEDENT
else : NEW_LINE INDENT print ( - 1 ) NEW_LINE return 0 NEW_LINE DEDENT
for i in range ( 50 ) : NEW_LINE INDENT if ( b [ i ] ) : NEW_LINE INDENT print ( i , end = " , " ) NEW_LINE DEDENT DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 3 NEW_LINE K = 40 NEW_LINE PowerArray ( N , K ) NEW_LINE DEDENT
from math import pow NEW_LINE
def findSum ( N , k ) : NEW_LINE
sum = 0 NEW_LINE for i in range ( 1 , N + 1 , 1 ) : NEW_LINE
sum += pow ( i , k ) NEW_LINE
return sum NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 8 NEW_LINE k = 4 NEW_LINE DEDENT
print ( int ( findSum ( N , k ) ) ) NEW_LINE
def countIndices ( arr , n ) : NEW_LINE
cnt = 0 ; NEW_LINE
max = 0 ; NEW_LINE for i in range ( n ) : NEW_LINE
if ( max < arr [ i ] ) : NEW_LINE
max = arr [ i ] ; NEW_LINE
cnt += 1 ; NEW_LINE return cnt ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 1 , 2 , 3 , 4 ] ; NEW_LINE n = len ( arr ) ; NEW_LINE print ( countIndices ( arr , n ) ) ; NEW_LINE DEDENT
bin = [ "000" , "001" , "010" , "011" , "100" , "101" , "110" , "111" ] ; NEW_LINE
def maxFreq ( s ) : NEW_LINE
binary = " " ; NEW_LINE
for i in range ( len ( s ) ) : NEW_LINE INDENT binary += bin [ ord ( s [ i ] ) - ord ( '0' ) ] ; NEW_LINE DEDENT
binary = binary [ 0 : len ( binary ) - 1 ] ; NEW_LINE count = 1 ; prev = - 1 ; j = 0 ; NEW_LINE for i in range ( len ( binary ) - 1 , - 1 , - 1 ) : NEW_LINE
if ( binary [ i ] == '1' ) : NEW_LINE
count = max ( count , j - prev ) ; NEW_LINE prev = j ; NEW_LINE j += 1 ; NEW_LINE return count ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT octal = "13" ; NEW_LINE print ( maxFreq ( octal ) ) ; NEW_LINE DEDENT
from math import sqrt , pow NEW_LINE sz = 100005 NEW_LINE isPrime = [ True for i in range ( sz + 1 ) ] NEW_LINE
def sieve ( ) : NEW_LINE INDENT isPrime [ 0 ] = isPrime [ 1 ] = False NEW_LINE for i in range ( 2 , int ( sqrt ( sz ) ) + 1 , 1 ) : NEW_LINE INDENT if ( isPrime [ i ] ) : NEW_LINE INDENT for j in range ( i * i , sz , i ) : NEW_LINE INDENT isPrime [ j ] = False NEW_LINE DEDENT DEDENT DEDENT DEDENT
def findPrimesD ( d ) : NEW_LINE
left = int ( pow ( 10 , d - 1 ) ) NEW_LINE right = int ( pow ( 10 , d ) - 1 ) NEW_LINE
for i in range ( left , right + 1 , 1 ) : NEW_LINE
if ( isPrime [ i ] ) : NEW_LINE INDENT print ( i , end = " ▁ " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE
sieve ( ) NEW_LINE d = 1 NEW_LINE findPrimesD ( d ) NEW_LINE
def Cells ( n , x ) : NEW_LINE INDENT if ( n <= 0 or x <= 0 or x > n * n ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT i = 1 NEW_LINE count = 0 NEW_LINE while ( i * i < x ) : NEW_LINE INDENT if ( x % i == 0 and x <= n * i ) : NEW_LINE INDENT count += 2 ; NEW_LINE DEDENT i += 1 NEW_LINE DEDENT if ( i * i == x ) : NEW_LINE INDENT return count + 1 NEW_LINE DEDENT else : NEW_LINE INDENT return count NEW_LINE DEDENT DEDENT
n = 6 NEW_LINE x = 12 NEW_LINE
print ( Cells ( n , x ) ) NEW_LINE
def maxOfMin ( a , n , S ) : NEW_LINE
mi = 10 ** 9 NEW_LINE
s1 = 0 NEW_LINE for i in range ( n ) : NEW_LINE INDENT s1 += a [ i ] NEW_LINE mi = min ( a [ i ] , mi ) NEW_LINE DEDENT
if ( s1 < S ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
if ( s1 == S ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
low = 0 NEW_LINE
high = mi NEW_LINE
ans = 0 NEW_LINE
while ( low <= high ) : NEW_LINE INDENT mid = ( low + high ) // 2 NEW_LINE DEDENT
if ( s1 - ( mid * n ) >= S ) : NEW_LINE INDENT ans = mid NEW_LINE low = mid + 1 NEW_LINE DEDENT
else : NEW_LINE INDENT high = mid - 1 NEW_LINE DEDENT
return ans NEW_LINE
a = [ 10 , 10 , 10 , 10 , 10 ] NEW_LINE S = 10 NEW_LINE n = len ( a ) NEW_LINE print ( maxOfMin ( a , n , S ) ) NEW_LINE
def Alphabet_N_Pattern ( N ) : NEW_LINE
Right = 1 NEW_LINE Left = 1 NEW_LINE Diagonal = 2 NEW_LINE
for index in range ( N ) : NEW_LINE
print ( Left , end = " " ) NEW_LINE Left += 1 NEW_LINE
for side_index in range ( 0 , 2 * ( index ) , 1 ) : NEW_LINE INDENT print ( " ▁ " , end = " " ) NEW_LINE DEDENT
if ( index != 0 and index != N - 1 ) : NEW_LINE INDENT print ( Diagonal , end = " " ) NEW_LINE Diagonal += 1 NEW_LINE DEDENT else : NEW_LINE INDENT print ( " ▁ " , end = " " ) NEW_LINE DEDENT
for side_index in range ( 0 , 2 * ( N - index - 1 ) , 1 ) : NEW_LINE INDENT print ( " ▁ " , end = " " ) NEW_LINE DEDENT
print ( Right , end = " " ) NEW_LINE Right += 1 NEW_LINE print ( " " , ▁ end ▁ = ▁ " " ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
Size = 6 NEW_LINE
Alphabet_N_Pattern ( Size ) NEW_LINE
def isSumDivides ( N ) : NEW_LINE INDENT temp = N NEW_LINE sum = 0 NEW_LINE DEDENT
while ( temp ) : NEW_LINE INDENT sum += temp % 10 NEW_LINE temp = int ( temp / 10 ) NEW_LINE DEDENT if ( N % sum == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT else : NEW_LINE INDENT return 0 NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 12 NEW_LINE if ( isSumDivides ( N ) ) : NEW_LINE INDENT print ( " YES " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NO " ) NEW_LINE DEDENT DEDENT
def sum ( N ) : NEW_LINE INDENT global S1 , S2 , S3 NEW_LINE S1 = ( ( ( N // 3 ) ) * ( 2 * 3 + ( N // 3 - 1 ) * 3 ) // 2 ) NEW_LINE S2 = ( ( ( N // 4 ) ) * ( 2 * 4 + ( N // 4 - 1 ) * 4 ) // 2 ) NEW_LINE S3 = ( ( ( N // 12 ) ) * ( 2 * 12 + ( N // 12 - 1 ) * 12 ) // 2 ) NEW_LINE return int ( S1 + S2 - S3 ) NEW_LINE DEDENT
/ * Driver code * / NEW_LINE if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 12 NEW_LINE print ( sum ( N ) ) NEW_LINE DEDENT
def nextGreater ( N ) : NEW_LINE INDENT power_of_2 = 1 ; NEW_LINE shift_count = 0 ; NEW_LINE DEDENT
while ( True ) : NEW_LINE
if ( ( ( N >> shift_count ) & 1 ) % 2 == 0 ) : NEW_LINE INDENT break ; NEW_LINE DEDENT
shift_count += 1 ; NEW_LINE
power_of_2 = power_of_2 * 2 ; NEW_LINE
return ( N + power_of_2 ) ; NEW_LINE
N = 11 ; NEW_LINE
print ( " The ▁ next ▁ number ▁ is ▁ = " , nextGreater ( N ) ) ; NEW_LINE
def countWays ( n ) : NEW_LINE
if ( n == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT if ( n <= 2 ) : NEW_LINE INDENT return n NEW_LINE DEDENT
f0 = 1 NEW_LINE f1 = 1 NEW_LINE f2 = 2 NEW_LINE ans = 0 NEW_LINE
for i in range ( 3 , n + 1 ) : NEW_LINE INDENT ans = f0 + f1 + f2 NEW_LINE f0 = f1 NEW_LINE f1 = f2 NEW_LINE f2 = ans NEW_LINE DEDENT
return ans NEW_LINE
n = 4 NEW_LINE print ( countWays ( n ) ) NEW_LINE
import numpy as np NEW_LINE n = 6 ; m = 6 ; NEW_LINE
def maxSum ( arr ) : NEW_LINE
dp = np . zeros ( ( n + 1 , 3 ) ) ; NEW_LINE
for i in range ( n ) : NEW_LINE
m1 = 0 ; m2 = 0 ; m3 = 0 ; NEW_LINE for j in range ( m ) : NEW_LINE
if ( ( j // ( m // 3 ) ) == 0 ) : NEW_LINE INDENT m1 = max ( m1 , arr [ i ] [ j ] ) ; NEW_LINE DEDENT
elif ( ( j // ( m // 3 ) ) == 1 ) : NEW_LINE INDENT m2 = max ( m2 , arr [ i ] [ j ] ) ; NEW_LINE DEDENT
elif ( ( j // ( m // 3 ) ) == 2 ) : NEW_LINE INDENT m3 = max ( m3 , arr [ i ] [ j ] ) ; NEW_LINE DEDENT
dp [ i + 1 ] [ 0 ] = max ( dp [ i ] [ 1 ] , dp [ i ] [ 2 ] ) + m1 ; NEW_LINE dp [ i + 1 ] [ 1 ] = max ( dp [ i ] [ 0 ] , dp [ i ] [ 2 ] ) + m2 ; NEW_LINE dp [ i + 1 ] [ 2 ] = max ( dp [ i ] [ 1 ] , dp [ i ] [ 0 ] ) + m3 ; NEW_LINE
print ( max ( max ( dp [ n ] [ 0 ] , dp [ n ] [ 1 ] ) , dp [ n ] [ 2 ] ) ) ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ [ 1 , 3 , 5 , 2 , 4 , 6 ] , [ 6 , 4 , 5 , 1 , 3 , 2 ] , [ 1 , 3 , 5 , 2 , 4 , 6 ] , [ 6 , 4 , 5 , 1 , 3 , 2 ] , [ 6 , 4 , 5 , 1 , 3 , 2 ] , [ 1 , 3 , 5 , 2 , 4 , 6 ] ] ; NEW_LINE maxSum ( arr ) ; NEW_LINE DEDENT
def solve ( s ) : NEW_LINE INDENT n = len ( s ) NEW_LINE DEDENT
dp = [ [ 0 for i in range ( n ) ] for i in range ( n ) ] NEW_LINE
for Len in range ( n - 1 , - 1 , - 1 ) : NEW_LINE
for i in range ( n ) : NEW_LINE INDENT if i + Len >= n : NEW_LINE INDENT break NEW_LINE DEDENT DEDENT
j = i + Len NEW_LINE
if ( i == 0 and j == n - 1 ) : NEW_LINE INDENT if ( s [ i ] == s [ j ] ) : NEW_LINE INDENT dp [ i ] [ j ] = 2 NEW_LINE DEDENT elif ( s [ i ] != s [ j ] ) : NEW_LINE INDENT dp [ i ] [ j ] = 1 NEW_LINE DEDENT DEDENT else : NEW_LINE INDENT if ( s [ i ] == s [ j ] ) : NEW_LINE DEDENT
if ( i - 1 >= 0 ) : NEW_LINE INDENT dp [ i ] [ j ] += dp [ i - 1 ] [ j ] NEW_LINE DEDENT if ( j + 1 <= n - 1 ) : NEW_LINE INDENT dp [ i ] [ j ] += dp [ i ] [ j + 1 ] NEW_LINE DEDENT if ( i - 1 < 0 or j + 1 >= n ) : NEW_LINE
dp [ i ] [ j ] += 1 NEW_LINE elif ( s [ i ] != s [ j ] ) : NEW_LINE
if ( i - 1 >= 0 ) : NEW_LINE INDENT dp [ i ] [ j ] += dp [ i - 1 ] [ j ] NEW_LINE DEDENT if ( j + 1 <= n - 1 ) : NEW_LINE INDENT dp [ i ] [ j ] += dp [ i ] [ j + 1 ] NEW_LINE DEDENT if ( i - 1 >= 0 and j + 1 <= n - 1 ) : NEW_LINE
dp [ i ] [ j ] -= dp [ i - 1 ] [ j + 1 ] NEW_LINE ways = [ ] NEW_LINE for i in range ( n ) : NEW_LINE if ( i == 0 or i == n - 1 ) : NEW_LINE
ways . append ( 1 ) NEW_LINE else : NEW_LINE
total = dp [ i - 1 ] [ i + 1 ] NEW_LINE ways . append ( total ) NEW_LINE for i in ways : NEW_LINE print ( i , end = " ▁ " ) NEW_LINE
s = " xyxyx " NEW_LINE solve ( s ) NEW_LINE
def getChicks ( n ) : NEW_LINE
size = max ( n , 7 ) ; NEW_LINE dp = [ 0 ] * size ; NEW_LINE dp [ 0 ] = 0 ; NEW_LINE dp [ 1 ] = 1 ; NEW_LINE
for i in range ( 2 , 7 ) : NEW_LINE INDENT dp [ i ] = dp [ i - 1 ] * 3 ; NEW_LINE DEDENT
dp [ 6 ] = 726 ; NEW_LINE
for i in range ( 8 , n + 1 ) : NEW_LINE
dp [ i ] = ( dp [ i - 1 ] - ( 2 * dp [ i - 6 ] // 3 ) ) * 3 ; NEW_LINE return dp [ n ] ; NEW_LINE
n = 3 ; NEW_LINE print ( getChicks ( n ) ) ; NEW_LINE
def getChicks ( n ) : NEW_LINE INDENT chicks = pow ( 3 , n - 1 ) NEW_LINE return chicks NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 3 NEW_LINE print ( getChicks ( n ) ) NEW_LINE DEDENT
import numpy as np NEW_LINE n = 3 NEW_LINE
dp = np . zeros ( ( n , n ) ) NEW_LINE
v = np . zeros ( ( n , n ) ) ; NEW_LINE
def minSteps ( i , j , arr ) : NEW_LINE
if ( i == n - 1 and j == n - 1 ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT if ( i > n - 1 or j > n - 1 ) : NEW_LINE INDENT return 9999999 ; NEW_LINE DEDENT
if ( v [ i ] [ j ] ) : NEW_LINE INDENT return dp [ i ] [ j ] ; NEW_LINE DEDENT v [ i ] [ j ] = 1 ; NEW_LINE dp [ i ] [ j ] = 9999999 ; NEW_LINE
for k in range ( max ( 0 , arr [ i ] [ j ] + j - n + 1 ) , min ( n - i - 1 , arr [ i ] [ j ] ) + 1 ) : NEW_LINE INDENT dp [ i ] [ j ] = min ( dp [ i ] [ j ] , minSteps ( i + k , j + arr [ i ] [ j ] - k , arr ) ) ; NEW_LINE DEDENT dp [ i ] [ j ] += 1 ; NEW_LINE return dp [ i ] [ j ] ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ [ 4 , 1 , 2 ] , [ 1 , 1 , 1 ] , [ 2 , 1 , 1 ] ] ; NEW_LINE ans = minSteps ( 0 , 0 , arr ) ; NEW_LINE if ( ans >= 9999999 ) : NEW_LINE INDENT print ( - 1 ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( ans ) ; NEW_LINE DEDENT DEDENT
import numpy as np ; NEW_LINE n = 3 NEW_LINE
dp = np . zeros ( ( n , n ) ) ; NEW_LINE
v = np . zeros ( ( n , n ) ) ; NEW_LINE
def minSteps ( i , j , arr ) : NEW_LINE
if ( i == n - 1 and j == n - 1 ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT if ( i > n - 1 or j > n - 1 ) : NEW_LINE INDENT return 9999999 ; NEW_LINE DEDENT
if ( v [ i ] [ j ] ) : NEW_LINE INDENT return dp [ i ] [ j ] ; NEW_LINE DEDENT v [ i ] [ j ] = 1 ; NEW_LINE
dp [ i ] [ j ] = 1 + min ( minSteps ( i + arr [ i ] [ j ] , j , arr ) , minSteps ( i , j + arr [ i ] [ j ] , arr ) ) ; NEW_LINE return dp [ i ] [ j ] ; NEW_LINE
arr = [ [ 2 , 1 , 2 ] , [ 1 , 1 , 1 ] , [ 1 , 1 , 1 ] ] ; NEW_LINE ans = minSteps ( 0 , 0 , arr ) ; NEW_LINE if ( ans >= 9999999 ) : NEW_LINE INDENT print ( - 1 ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( ans ) ; NEW_LINE DEDENT
MAX = 1001 NEW_LINE dp = [ [ - 1 for i in range ( MAX ) ] for i in range ( MAX ) ] NEW_LINE
def MaxProfit ( treasure , color , n , k , col , A , B ) : NEW_LINE INDENT if ( k == n ) : NEW_LINE DEDENT
dp [ k ] [ col ] = 0 NEW_LINE return dp [ k ] [ col ] NEW_LINE if ( dp [ k ] [ col ] != - 1 ) : NEW_LINE return dp [ k ] [ col ] NEW_LINE summ = 0 NEW_LINE
summ += max ( A * treasure [ k ] + MaxProfit ( treasure , color , n , k + 1 , color [ k ] , A , B ) , MaxProfit ( treasure , color , n , k + 1 , col , A , B ) ) NEW_LINE else : NEW_LINE summ += max ( B * treasure [ k ] + MaxProfit ( treasure , color , n , k + 1 , color [ k ] , A , B ) , MaxProfit ( treasure , color , n , k + 1 , col , A , B ) ) NEW_LINE dp [ k ] [ col ] = summ NEW_LINE
return dp [ k ] [ col ] NEW_LINE
A = - 5 NEW_LINE B = 7 NEW_LINE treasure = [ 4 , 8 , 2 , 9 ] NEW_LINE color = [ 2 , 2 , 6 , 2 ] NEW_LINE n = len ( color ) NEW_LINE print ( MaxProfit ( treasure , color , n , 0 , 0 , A , B ) ) NEW_LINE
def printTetra ( n ) : NEW_LINE INDENT dp = [ 0 ] * ( n + 5 ) ; NEW_LINE DEDENT
dp [ 0 ] = 0 ; NEW_LINE dp [ 1 ] = 1 ; NEW_LINE dp [ 2 ] = 1 ; NEW_LINE dp [ 3 ] = 2 ; NEW_LINE for i in range ( 4 , n + 1 ) : NEW_LINE INDENT dp [ i ] = ( dp [ i - 1 ] + dp [ i - 2 ] + dp [ i - 3 ] + dp [ i - 4 ] ) ; NEW_LINE DEDENT print ( dp [ n ] ) ; NEW_LINE
n = 10 ; NEW_LINE printTetra ( n ) ; NEW_LINE
def maxSum1 ( arr , n ) : NEW_LINE INDENT dp = [ 0 ] * n NEW_LINE maxi = 0 NEW_LINE for i in range ( n - 1 ) : NEW_LINE DEDENT
dp [ i ] = arr [ i ] NEW_LINE
if ( maxi < arr [ i ] ) : NEW_LINE INDENT maxi = arr [ i ] NEW_LINE DEDENT
for i in range ( 2 , n - 1 ) : NEW_LINE
for j in range ( i - 1 ) : NEW_LINE
if ( dp [ i ] < dp [ j ] + arr [ i ] ) : NEW_LINE INDENT dp [ i ] = dp [ j ] + arr [ i ] NEW_LINE DEDENT
if ( maxi < dp [ i ] ) : NEW_LINE INDENT maxi = dp [ i ] NEW_LINE DEDENT
return maxi NEW_LINE
def maxSum2 ( arr , n ) : NEW_LINE INDENT dp = [ 0 ] * n NEW_LINE maxi = 0 NEW_LINE for i in range ( 1 , n ) : NEW_LINE INDENT dp [ i ] = arr [ i ] NEW_LINE if ( maxi < arr [ i ] ) : NEW_LINE INDENT maxi = arr [ i ] NEW_LINE DEDENT DEDENT DEDENT
for i in range ( 3 , n ) : NEW_LINE
for j in range ( 1 , i - 1 ) : NEW_LINE
if ( dp [ i ] < arr [ i ] + dp [ j ] ) : NEW_LINE INDENT dp [ i ] = arr [ i ] + dp [ j ] NEW_LINE DEDENT
if ( maxi < dp [ i ] ) : NEW_LINE INDENT maxi = dp [ i ] NEW_LINE DEDENT
return maxi NEW_LINE def findMaxSum ( arr , n ) : NEW_LINE return max ( maxSum1 ( arr , n ) , maxSum2 ( arr , n ) ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 1 , 2 , 3 , 1 ] NEW_LINE n = len ( arr ) NEW_LINE print ( findMaxSum ( arr , n ) ) NEW_LINE DEDENT
def permutationCoeff ( n , k ) : NEW_LINE INDENT P = [ [ 0 for i in range ( k + 1 ) ] for j in range ( n + 1 ) ] NEW_LINE DEDENT
for i in range ( n + 1 ) : NEW_LINE INDENT for j in range ( min ( i , k ) + 1 ) : NEW_LINE DEDENT
if ( j == 0 ) : NEW_LINE INDENT P [ i ] [ j ] = 1 NEW_LINE DEDENT
else : NEW_LINE INDENT P [ i ] [ j ] = P [ i - 1 ] [ j ] + ( j * P [ i - 1 ] [ j - 1 ] ) NEW_LINE DEDENT
if ( j < k ) : NEW_LINE INDENT P [ i ] [ j + 1 ] = 0 NEW_LINE DEDENT return P [ n ] [ k ] NEW_LINE
n = 10 NEW_LINE k = 2 NEW_LINE print ( " Value ▁ fo ▁ P ( " , n , " , ▁ " , k , " ) ▁ is ▁ " , permutationCoeff ( n , k ) , sep = " " ) NEW_LINE
def permutationCoeff ( n , k ) : NEW_LINE INDENT fact = [ 0 for i in range ( n + 1 ) ] NEW_LINE DEDENT
fact [ 0 ] = 1 NEW_LINE
for i in range ( 1 , n + 1 ) : NEW_LINE INDENT fact [ i ] = i * fact [ i - 1 ] NEW_LINE DEDENT
return int ( fact [ n ] / fact [ n - k ] ) NEW_LINE
n = 10 NEW_LINE k = 2 NEW_LINE print ( " Value ▁ of ▁ P ( " , n , " , ▁ " , k , " ) ▁ is ▁ " , permutationCoeff ( n , k ) , sep = " " ) NEW_LINE
def isSubsetSum ( set , n , sum ) : NEW_LINE
if ( sum == 0 ) : NEW_LINE INDENT return True NEW_LINE DEDENT if ( n == 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT
if ( set [ n - 1 ] > sum ) : NEW_LINE INDENT return isSubsetSum ( set , n - 1 , sum ) NEW_LINE DEDENT
return isSubsetSum ( set , n - 1 , sum ) or isSubsetSum ( set , n - 1 , sum - set [ n - 1 ] ) NEW_LINE
set = [ 3 , 34 , 4 , 12 , 5 , 2 ] NEW_LINE sum = 9 NEW_LINE n = len ( set ) NEW_LINE if ( isSubsetSum ( set , n , sum ) == True ) : NEW_LINE INDENT print ( " Found ▁ a ▁ subset ▁ with ▁ given ▁ sum " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No ▁ subset ▁ with ▁ given ▁ sum " ) NEW_LINE DEDENT
def compute_z ( s , z ) : NEW_LINE INDENT l = 0 NEW_LINE r = 0 NEW_LINE n = len ( s ) NEW_LINE for i in range ( 1 , n , 1 ) : NEW_LINE INDENT if ( i > r ) : NEW_LINE INDENT l = i NEW_LINE r = i NEW_LINE while ( r < n and s [ r - l ] == s [ r ] ) : NEW_LINE INDENT r += 1 NEW_LINE DEDENT z [ i ] = r - l NEW_LINE r -= 1 NEW_LINE DEDENT else : NEW_LINE INDENT k = i - l NEW_LINE if ( z [ k ] < r - i + 1 ) : NEW_LINE INDENT z [ i ] = z [ k ] NEW_LINE DEDENT else : NEW_LINE INDENT l = i NEW_LINE while ( r < n and s [ r - l ] == s [ r ] ) : NEW_LINE INDENT r += 1 NEW_LINE DEDENT z [ i ] = r - l NEW_LINE r -= 1 NEW_LINE DEDENT DEDENT DEDENT DEDENT
def countPermutation ( a , b ) : NEW_LINE
b = b + b NEW_LINE
b = b [ 0 : len ( b ) - 1 ] NEW_LINE
ans = 0 NEW_LINE s = a + " $ " + b NEW_LINE n = len ( s ) NEW_LINE
z = [ 0 for i in range ( n ) ] NEW_LINE compute_z ( s , z ) NEW_LINE for i in range ( 1 , n , 1 ) : NEW_LINE
if ( z [ i ] == len ( a ) ) : NEW_LINE INDENT ans += 1 NEW_LINE DEDENT return ans NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT a = "101" NEW_LINE b = "101" NEW_LINE print ( countPermutation ( a , b ) ) NEW_LINE DEDENT
def smallestSubsequence ( S , K ) : NEW_LINE
N = len ( S ) NEW_LINE
answer = [ ] NEW_LINE
for i in range ( N ) : NEW_LINE
if ( len ( answer ) == 0 ) : NEW_LINE INDENT answer . append ( S [ i ] ) NEW_LINE DEDENT else : NEW_LINE
while ( len ( answer ) > 0 and ( S [ i ] < answer [ len ( answer ) - 1 ] ) and ( len ( answer ) - 1 + N - i >= K ) ) : NEW_LINE INDENT answer = answer [ : - 1 ] NEW_LINE DEDENT
if ( len ( answer ) == 0 or len ( answer ) < K ) : NEW_LINE
answer . append ( S [ i ] ) NEW_LINE
ret = [ ] NEW_LINE
while ( len ( answer ) > 0 ) : NEW_LINE INDENT ret . append ( answer [ len ( answer ) - 1 ] ) NEW_LINE answer = answer [ : - 1 ] NEW_LINE DEDENT
ret = ret [ : : - 1 ] NEW_LINE ret = ' ' . join ( ret ) NEW_LINE
print ( ret ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT S = " aabdaabc " NEW_LINE K = 3 NEW_LINE smallestSubsequence ( S , K ) NEW_LINE DEDENT
from math import sqrt , floor , ceil NEW_LINE
def is_rtol ( s ) : NEW_LINE INDENT tmp = floor ( sqrt ( len ( s ) ) ) - 1 NEW_LINE first = s [ tmp ] NEW_LINE DEDENT
for pos in range ( tmp , len ( s ) - 1 , tmp ) : NEW_LINE
if ( s [ pos ] != first ) : NEW_LINE INDENT return False NEW_LINE DEDENT return True NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
str = " abcxabxcaxbcxabc " NEW_LINE
if ( is_rtol ( str ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
def check ( str , K ) : NEW_LINE
if ( len ( str ) % K == 0 ) : NEW_LINE INDENT sum = 0 NEW_LINE DEDENT
for i in range ( K ) : NEW_LINE INDENT sum += ord ( str [ i ] ) ; NEW_LINE DEDENT
for j in range ( K , len ( str ) , K ) : NEW_LINE INDENT s_comp = 0 ; NEW_LINE for p in range ( j , j + K ) : NEW_LINE INDENT s_comp += ord ( str [ p ] ) ; NEW_LINE DEDENT DEDENT
if ( s_comp != sum ) : NEW_LINE
return False ; NEW_LINE
return True ; NEW_LINE
return False ; NEW_LINE
K = 3 ; NEW_LINE str = " abdcbbdba " ; NEW_LINE if ( check ( str , K ) ) : NEW_LINE INDENT print ( " YES " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NO " ) NEW_LINE DEDENT
def maxSum ( str ) : NEW_LINE INDENT maximumSum = 0 NEW_LINE DEDENT
totalOnes = 0 NEW_LINE for i in str : NEW_LINE INDENT if i == '1' : NEW_LINE INDENT totalOnes += 1 NEW_LINE DEDENT DEDENT
zero = 0 NEW_LINE ones = 0 NEW_LINE
i = 0 NEW_LINE while i < len ( str ) : NEW_LINE INDENT if ( str [ i ] == '0' ) : NEW_LINE INDENT zero += 1 NEW_LINE DEDENT else : NEW_LINE INDENT ones += 1 NEW_LINE DEDENT DEDENT
maximumSum = max ( maximumSum , zero + ( totalOnes - ones ) ) NEW_LINE i += 1 NEW_LINE return maximumSum NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
str = "011101" NEW_LINE
print ( maxSum ( str ) ) NEW_LINE
def maxLenSubStr ( s ) : NEW_LINE
if ( len ( s ) < 3 ) : NEW_LINE INDENT return len ( s ) NEW_LINE DEDENT
temp = 2 NEW_LINE ans = 2 NEW_LINE
for i in range ( 2 , len ( s ) ) : NEW_LINE
if ( s [ i ] != s [ i - 1 ] or s [ i ] != s [ i - 2 ] ) : NEW_LINE INDENT temp += 1 NEW_LINE DEDENT
else : NEW_LINE INDENT ans = max ( temp , ans ) NEW_LINE temp = 2 NEW_LINE DEDENT ans = max ( temp , ans ) NEW_LINE return ans NEW_LINE
s = " baaabbabbb " NEW_LINE print ( maxLenSubStr ( s ) ) NEW_LINE
def no_of_ways ( s ) : NEW_LINE INDENT n = len ( s ) NEW_LINE DEDENT
count_left = 0 NEW_LINE count_right = 0 NEW_LINE
for i in range ( 0 , n , 1 ) : NEW_LINE INDENT if ( s [ i ] == s [ 0 ] ) : NEW_LINE INDENT count_left += 1 NEW_LINE DEDENT else : NEW_LINE INDENT break NEW_LINE DEDENT DEDENT
i = n - 1 NEW_LINE while ( i >= 0 ) : NEW_LINE INDENT if ( s [ i ] == s [ n - 1 ] ) : NEW_LINE INDENT count_right += 1 NEW_LINE DEDENT else : NEW_LINE INDENT break NEW_LINE DEDENT i -= 1 NEW_LINE DEDENT
if ( s [ 0 ] == s [ n - 1 ] ) : NEW_LINE INDENT return ( ( count_left + 1 ) * ( count_right + 1 ) ) NEW_LINE DEDENT
else : NEW_LINE INDENT return ( count_left + count_right + 1 ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT s = " geeksforgeeks " NEW_LINE print ( no_of_ways ( s ) ) NEW_LINE DEDENT
def preCompute ( n , s , pref ) : NEW_LINE INDENT for i in range ( 1 , n ) : NEW_LINE INDENT pref [ i ] = pref [ i - 1 ] NEW_LINE if s [ i - 1 ] == s [ i ] : NEW_LINE INDENT pref [ i ] += 1 NEW_LINE DEDENT DEDENT DEDENT
def query ( pref , l , r ) : NEW_LINE INDENT return pref [ r ] - pref [ l ] NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT s = " ggggggg " NEW_LINE n = len ( s ) NEW_LINE pref = [ 0 ] * n NEW_LINE preCompute ( n , s , pref ) NEW_LINE DEDENT
l = 1 NEW_LINE r = 2 NEW_LINE print ( query ( pref , l , r ) ) NEW_LINE
l = 1 NEW_LINE r = 5 NEW_LINE print ( query ( pref , l , r ) ) NEW_LINE
def findDirection ( s ) : NEW_LINE INDENT count = 0 NEW_LINE d = " " NEW_LINE for i in range ( len ( s ) ) : NEW_LINE INDENT if ( s [ i ] == ' L ' ) : NEW_LINE INDENT count -= 1 NEW_LINE DEDENT else : NEW_LINE INDENT if ( s [ i ] == ' R ' ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT DEDENT DEDENT DEDENT
if ( count > 0 ) : NEW_LINE INDENT if ( count % 4 == 0 ) : NEW_LINE INDENT d = " N " NEW_LINE DEDENT elif ( count % 4 == 10 ) : NEW_LINE INDENT d = " E " NEW_LINE DEDENT elif ( count % 4 == 2 ) : NEW_LINE INDENT d = " S " NEW_LINE DEDENT elif ( count % 4 == 3 ) : NEW_LINE INDENT d = " W " NEW_LINE DEDENT DEDENT
if ( count < 0 ) : NEW_LINE INDENT count *= - 1 NEW_LINE if ( count % 4 == 0 ) : NEW_LINE INDENT d = " N " NEW_LINE DEDENT elif ( count % 4 == 1 ) : NEW_LINE INDENT d = " W " NEW_LINE DEDENT elif ( count % 4 == 2 ) : NEW_LINE INDENT d = " S " NEW_LINE DEDENT elif ( count % 4 == 3 ) : NEW_LINE INDENT d = " E " NEW_LINE DEDENT DEDENT return d NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT s = " LLRLRRL " NEW_LINE print ( findDirection ( s ) ) NEW_LINE s = " LL " NEW_LINE print ( findDirection ( s ) ) NEW_LINE DEDENT
def isCheck ( str ) : NEW_LINE INDENT length = len ( str ) NEW_LINE lowerStr , upperStr = " " , " " NEW_LINE DEDENT
for i in range ( length ) : NEW_LINE
if ( ord ( str [ i ] ) >= 65 and ord ( str [ i ] ) <= 91 ) : NEW_LINE INDENT upperStr = upperStr + str [ i ] NEW_LINE DEDENT else : NEW_LINE INDENT lowerStr = lowerStr + str [ i ] NEW_LINE DEDENT
transformStr = lowerStr . upper ( ) NEW_LINE return transformStr == upperStr NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT str = " geeGkEEsKS " NEW_LINE if isCheck ( str ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT DEDENT
def encode ( s , k ) : NEW_LINE
newS = " " NEW_LINE
for i in range ( len ( s ) ) : NEW_LINE
val = ord ( s [ i ] ) NEW_LINE
dup = k NEW_LINE
' NEW_LINE INDENT if val + k > 122 : NEW_LINE INDENT k -= ( 122 - val ) NEW_LINE k = k % 26 NEW_LINE newS += chr ( 96 + k ) NEW_LINE DEDENT else : NEW_LINE INDENT newS += chr ( val + k ) NEW_LINE DEDENT k = dup NEW_LINE DEDENT
print ( newS ) NEW_LINE
str = " abc " NEW_LINE k = 28 NEW_LINE
encode ( str , k ) NEW_LINE
def isVowel ( x ) : NEW_LINE INDENT if ( x == ' a ' or x == ' e ' or x == ' i ' or x == ' o ' or x == ' u ' ) : NEW_LINE INDENT return True NEW_LINE DEDENT else : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT
def updateSandwichedVowels ( a ) : NEW_LINE INDENT n = len ( a ) NEW_LINE DEDENT
updatedString = " " NEW_LINE
for i in range ( 0 , n , 1 ) : NEW_LINE
' NEW_LINE INDENT if ( i == 0 or i == n - 1 ) : NEW_LINE INDENT updatedString += a [ i ] NEW_LINE continue NEW_LINE DEDENT DEDENT
if ( isVowel ( a [ i ] ) == True and isVowel ( a [ i - 1 ] ) == False and isVowel ( a [ i + 1 ] ) == False ) : NEW_LINE INDENT continue NEW_LINE DEDENT
updatedString += a [ i ] NEW_LINE return updatedString NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT str = " geeksforgeeks " NEW_LINE DEDENT
updatedString = updateSandwichedVowels ( str ) NEW_LINE print ( updatedString ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def findPathUtil ( root : Node , k : int , path : list , flag : int ) : NEW_LINE INDENT global ans NEW_LINE if root is None : NEW_LINE INDENT return NEW_LINE DEDENT DEDENT
if root . data >= k : NEW_LINE INDENT flag = 1 NEW_LINE DEDENT
if root . left is None and root . right is None : NEW_LINE INDENT if flag : NEW_LINE INDENT ans = 1 NEW_LINE print ( " ( " , end = " " ) NEW_LINE for i in range ( len ( path ) ) : NEW_LINE INDENT print ( path [ i ] , end = " , ▁ " ) NEW_LINE DEDENT print ( root . data , end = " ) , ▁ " ) NEW_LINE DEDENT return NEW_LINE DEDENT
path . append ( root . data ) NEW_LINE
findPathUtil ( root . left , k , path , flag ) NEW_LINE findPathUtil ( root . right , k , path , flag ) NEW_LINE
path . pop ( ) NEW_LINE
def findPath ( root : Node , k : int ) : NEW_LINE INDENT global ans NEW_LINE DEDENT
flag = 0 NEW_LINE
ans = 0 NEW_LINE v = [ ] NEW_LINE
findPathUtil ( root , k , v , flag ) NEW_LINE
if ans == 0 : NEW_LINE INDENT print ( - 1 ) NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT ans = 0 NEW_LINE k = 25 NEW_LINE DEDENT
root = Node ( 10 ) NEW_LINE root . left = Node ( 5 ) NEW_LINE root . right = Node ( 8 ) NEW_LINE root . left . left = Node ( 29 ) NEW_LINE root . left . right = Node ( 2 ) NEW_LINE root . right . right = Node ( 98 ) NEW_LINE root . right . left = Node ( 1 ) NEW_LINE root . right . right . right = Node ( 50 ) NEW_LINE root . left . left . left = Node ( 20 ) NEW_LINE findPath ( root , k ) NEW_LINE
def Tridecagonal_num ( n ) : NEW_LINE
return ( 11 * n * n - 9 * n ) / 2 NEW_LINE
n = 3 NEW_LINE print ( int ( Tridecagonal_num ( n ) ) ) NEW_LINE n = 10 NEW_LINE print ( int ( Tridecagonal_num ( n ) ) ) NEW_LINE
def findNumbers ( n , w ) : NEW_LINE INDENT x = 0 ; NEW_LINE sum = 0 ; NEW_LINE DEDENT
if ( w >= 0 and w <= 8 ) : NEW_LINE
x = 9 - w ; NEW_LINE
elif ( w >= - 9 and w <= - 1 ) : NEW_LINE
x = 10 + w ; NEW_LINE sum = pow ( 10 , n - 2 ) ; NEW_LINE sum = ( x * sum ) ; NEW_LINE return sum ; NEW_LINE
n = 3 ; NEW_LINE w = 4 ; NEW_LINE
print ( findNumbers ( n , w ) ) ; NEW_LINE
def MaximumHeight ( a , n ) : NEW_LINE INDENT result = 1 NEW_LINE for i in range ( 1 , n ) : NEW_LINE DEDENT
y = ( i * ( i + 1 ) ) / 2 NEW_LINE
if ( y < n ) : NEW_LINE INDENT result = i NEW_LINE DEDENT
else : NEW_LINE INDENT break NEW_LINE DEDENT return result NEW_LINE
arr = [ 40 , 100 , 20 , 30 ] NEW_LINE n = len ( arr ) NEW_LINE print ( MaximumHeight ( arr , n ) ) NEW_LINE
def findK ( n , k ) : NEW_LINE INDENT a = list ( ) NEW_LINE DEDENT
i = 1 NEW_LINE while i < n : NEW_LINE INDENT a . append ( i ) NEW_LINE i = i + 2 NEW_LINE DEDENT
i = 2 NEW_LINE while i < n : NEW_LINE INDENT a . append ( i ) NEW_LINE i = i + 2 NEW_LINE DEDENT return ( a [ k - 1 ] ) NEW_LINE
n = 10 NEW_LINE k = 3 NEW_LINE print ( findK ( n , k ) ) NEW_LINE
def factorial ( n ) : NEW_LINE
return 1 if ( n == 1 or n == 0 ) else n * factorial ( n - 1 ) ; NEW_LINE
num = 5 ; NEW_LINE print ( " Factorial ▁ of " , num , " is " , factorial ( num ) ) ; NEW_LINE
def pell ( n ) : NEW_LINE INDENT if ( n <= 2 ) : NEW_LINE INDENT return n NEW_LINE DEDENT a = 1 NEW_LINE b = 2 NEW_LINE for i in range ( 3 , n + 1 ) : NEW_LINE INDENT c = 2 * b + a NEW_LINE a = b NEW_LINE b = c NEW_LINE DEDENT return b NEW_LINE DEDENT
n = 4 NEW_LINE print ( pell ( n ) ) NEW_LINE
def isMultipleOf10 ( n ) : NEW_LINE INDENT return ( n % 15 == 0 ) NEW_LINE DEDENT
n = 30 NEW_LINE if ( isMultipleOf10 ( n ) ) : NEW_LINE INDENT print ( " Yes " ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) ; NEW_LINE DEDENT
def countOddPrimeFactors ( n ) : NEW_LINE INDENT result = 1 ; NEW_LINE DEDENT
while ( n % 2 == 0 ) : NEW_LINE INDENT n /= 2 NEW_LINE DEDENT
i = 3 NEW_LINE while i * i <= n : NEW_LINE INDENT divCount = 0 NEW_LINE DEDENT
while ( n % i == 0 ) : NEW_LINE INDENT n /= i NEW_LINE divCount = divCount + 1 NEW_LINE DEDENT result = result * divCount + 1 NEW_LINE i = i + 2 NEW_LINE
if ( n > 2 ) : NEW_LINE INDENT result = result * 2 NEW_LINE DEDENT return result NEW_LINE def politness ( n ) : NEW_LINE return countOddPrimeFactors ( n ) - 1 ; NEW_LINE
n = 90 NEW_LINE print " Politness ▁ of ▁ " , n , " ▁ = ▁ " , politness ( n ) NEW_LINE n = 15 NEW_LINE print " Politness ▁ of ▁ " , n , " ▁ = ▁ " , politness ( n ) NEW_LINE
import math NEW_LINE MAX = 10000 ; NEW_LINE
primes = [ ] ; NEW_LINE
def Sieve ( ) : NEW_LINE INDENT n = MAX ; NEW_LINE DEDENT
nNew = int ( math . sqrt ( n ) ) ; NEW_LINE
marked = [ 0 ] * ( int ( n / 2 + 500 ) ) ; NEW_LINE
for i in range ( 1 , int ( ( nNew - 1 ) / 2 ) + 1 ) : NEW_LINE INDENT for j in range ( ( ( i * ( i + 1 ) ) << 1 ) , ( int ( n / 2 ) + 1 ) , ( 2 * i + 1 ) ) : NEW_LINE INDENT marked [ j ] = 1 ; NEW_LINE DEDENT DEDENT
primes . append ( 2 ) ; NEW_LINE
for i in range ( 1 , int ( n / 2 ) + 1 ) : NEW_LINE INDENT if ( marked [ i ] == 0 ) : NEW_LINE INDENT primes . append ( 2 * i + 1 ) ; NEW_LINE DEDENT DEDENT
def binarySearch ( left , right , n ) : NEW_LINE INDENT if ( left <= right ) : NEW_LINE INDENT mid = int ( ( left + right ) / 2 ) ; NEW_LINE DEDENT DEDENT
if ( mid == 0 or mid == len ( primes ) - 1 ) : NEW_LINE INDENT return primes [ mid ] ; NEW_LINE DEDENT
if ( primes [ mid ] == n ) : NEW_LINE INDENT return primes [ mid - 1 ] ; NEW_LINE DEDENT
if ( primes [ mid ] < n and primes [ mid + 1 ] > n ) : NEW_LINE INDENT return primes [ mid ] ; NEW_LINE DEDENT if ( n < primes [ mid ] ) : NEW_LINE INDENT return binarySearch ( left , mid - 1 , n ) ; NEW_LINE DEDENT else : NEW_LINE INDENT return binarySearch ( mid + 1 , right , n ) ; NEW_LINE DEDENT return 0 ; NEW_LINE
Sieve ( ) ; NEW_LINE n = 17 ; NEW_LINE print ( binarySearch ( 0 , len ( primes ) - 1 , n ) ) ; NEW_LINE
def factorial ( n ) : NEW_LINE INDENT if n == 0 : NEW_LINE INDENT return 1 NEW_LINE DEDENT return n * factorial ( n - 1 ) NEW_LINE DEDENT
num = 5 ; NEW_LINE print ( " Factorial ▁ of " , num , " is " , factorial ( num ) ) NEW_LINE
def FlipBits ( n ) : NEW_LINE INDENT n -= ( n & ( - n ) ) ; NEW_LINE return n ; NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 12 ; NEW_LINE print ( " The ▁ number ▁ after ▁ unsetting ▁ the " , end = " " ) ; NEW_LINE print ( " ▁ rightmost ▁ set ▁ bit : ▁ " , FlipBits ( N ) ) ; NEW_LINE DEDENT
def Maximum_xor_Triplet ( n , a ) : NEW_LINE
s = set ( ) NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT for j in range ( i , n ) : NEW_LINE DEDENT
s . add ( a [ i ] ^ a [ j ] ) NEW_LINE ans = 0 NEW_LINE for i in s : NEW_LINE for j in range ( 0 , n ) : NEW_LINE
ans = max ( ans , i ^ a [ j ] ) NEW_LINE print ( ans ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = [ 1 , 3 , 8 , 15 ] NEW_LINE n = len ( a ) NEW_LINE Maximum_xor_Triplet ( n , a ) NEW_LINE DEDENT
from bisect import bisect_left NEW_LINE
def printMissing ( arr , n , low , high ) : NEW_LINE INDENT arr . sort ( ) NEW_LINE DEDENT
ptr = bisect_left ( arr , low ) NEW_LINE index = ptr NEW_LINE
i = index NEW_LINE x = low NEW_LINE while ( i < n and x <= high ) : NEW_LINE
if ( arr [ i ] != x ) : NEW_LINE INDENT print ( x , end = " ▁ " ) NEW_LINE DEDENT
else : NEW_LINE INDENT i = i + 1 NEW_LINE DEDENT
x = x + 1 NEW_LINE
while ( x <= high ) : NEW_LINE INDENT print ( x , end = " ▁ " ) NEW_LINE x = x + 1 NEW_LINE DEDENT
arr = [ 1 , 3 , 5 , 4 ] NEW_LINE n = len ( arr ) NEW_LINE low = 1 NEW_LINE high = 10 NEW_LINE printMissing ( arr , n , low , high ) ; NEW_LINE
def printMissing ( arr , n , low , high ) : NEW_LINE
points_of_range = [ False ] * ( high - low + 1 ) NEW_LINE for i in range ( n ) : NEW_LINE
if ( low <= arr [ i ] and arr [ i ] <= high ) : NEW_LINE INDENT points_of_range [ arr [ i ] - low ] = True NEW_LINE DEDENT
for x in range ( high - low + 1 ) : NEW_LINE INDENT if ( points_of_range [ x ] == False ) : NEW_LINE INDENT print ( low + x , end = " ▁ " ) NEW_LINE DEDENT DEDENT
arr = [ 1 , 3 , 5 , 4 ] NEW_LINE n = len ( arr ) NEW_LINE low , high = 1 , 10 NEW_LINE printMissing ( arr , n , low , high ) NEW_LINE
def printMissing ( arr , n , low , high ) : NEW_LINE
s = set ( arr ) NEW_LINE
for x in range ( low , high + 1 ) : NEW_LINE INDENT if x not in s : NEW_LINE INDENT print ( x , end = ' ▁ ' ) NEW_LINE DEDENT DEDENT
arr = [ 1 , 3 , 5 , 4 ] NEW_LINE n = len ( arr ) NEW_LINE low , high = 1 , 10 NEW_LINE printMissing ( arr , n , low , high ) NEW_LINE
def find ( a , b , k , n1 , n2 ) : NEW_LINE
s = set ( ) NEW_LINE for i in range ( n2 ) : NEW_LINE INDENT s . add ( b [ i ] ) NEW_LINE DEDENT
missing = 0 NEW_LINE for i in range ( n1 ) : NEW_LINE INDENT if a [ i ] not in s : NEW_LINE INDENT missing += 1 NEW_LINE DEDENT if missing == k : NEW_LINE INDENT return a [ i ] NEW_LINE DEDENT DEDENT return - 1 NEW_LINE
a = [ 0 , 2 , 4 , 6 , 8 , 10 , 12 , 14 , 15 ] NEW_LINE b = [ 4 , 10 , 6 , 8 , 12 ] NEW_LINE n1 = len ( a ) NEW_LINE n2 = len ( b ) NEW_LINE k = 3 NEW_LINE print ( find ( a , b , k , n1 , n2 ) ) NEW_LINE
def findString ( S , N ) : NEW_LINE
amounts = [ 0 ] * 26 NEW_LINE
for i in range ( len ( S ) ) : NEW_LINE INDENT amounts [ ord ( S [ i ] ) - 97 ] += 1 NEW_LINE DEDENT count = 0 NEW_LINE
for i in range ( 26 ) : NEW_LINE INDENT if amounts [ i ] > 0 : NEW_LINE INDENT count += 1 NEW_LINE DEDENT DEDENT
if count > N : NEW_LINE INDENT print ( " - 1" ) NEW_LINE DEDENT
else : NEW_LINE INDENT ans = " " NEW_LINE high = 100001 NEW_LINE low = 0 NEW_LINE DEDENT
while ( high - low ) > 1 : NEW_LINE INDENT total = 0 NEW_LINE DEDENT
mid = ( high + low ) // 2 NEW_LINE
for i in range ( 26 ) : NEW_LINE
if amounts [ i ] > 0 : NEW_LINE INDENT total += ( amounts [ i ] - 1 ) // mid + 1 NEW_LINE DEDENT
if total <= N : NEW_LINE INDENT high = mid NEW_LINE DEDENT else : NEW_LINE INDENT low = mid NEW_LINE DEDENT print ( high , end = " ▁ " ) NEW_LINE total = 0 NEW_LINE
for i in range ( 26 ) : NEW_LINE INDENT if amounts [ i ] > 0 : NEW_LINE INDENT total += ( amounts [ i ] - 1 ) // high + 1 NEW_LINE for j in range ( ( amounts [ i ] - 1 ) // high + 1 ) : NEW_LINE DEDENT DEDENT
ans += chr ( i + 97 ) NEW_LINE
for i in range ( total , N ) : NEW_LINE INDENT ans += ' a ' NEW_LINE DEDENT ans = ans [ : : - 1 ] NEW_LINE
print ( ans ) NEW_LINE
S = " toffee " NEW_LINE K = 4 NEW_LINE findString ( S , K ) NEW_LINE
def printFirstRepeating ( arr , n ) : NEW_LINE
Min = - 1 NEW_LINE
myset = dict ( ) NEW_LINE
for i in range ( n - 1 , - 1 , - 1 ) : NEW_LINE
if arr [ i ] in myset . keys ( ) : NEW_LINE INDENT Min = i NEW_LINE DEDENT
else : NEW_LINE INDENT myset [ arr [ i ] ] = 1 NEW_LINE DEDENT
if ( Min != - 1 ) : NEW_LINE INDENT print ( " The ▁ first ▁ repeating ▁ element ▁ is " , arr [ Min ] ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " There ▁ are ▁ no ▁ repeating ▁ elements " ) NEW_LINE DEDENT
arr = [ 10 , 5 , 3 , 4 , 3 , 5 , 6 ] NEW_LINE n = len ( arr ) NEW_LINE printFirstRepeating ( arr , n ) NEW_LINE
def printFirstRepeating ( arr , n ) : NEW_LINE
k = 0 NEW_LINE
max = n NEW_LINE for i in range ( n ) : NEW_LINE INDENT if ( max < arr [ i ] ) : NEW_LINE INDENT max = arr [ i ] NEW_LINE DEDENT DEDENT
a = [ 0 for i in range ( max + 1 ) ] NEW_LINE
b = [ 0 for i in range ( max + 1 ) ] NEW_LINE for i in range ( n ) : NEW_LINE
if ( a [ arr [ i ] ] ) : NEW_LINE INDENT b [ arr [ i ] ] = 1 NEW_LINE k = 1 NEW_LINE continue NEW_LINE DEDENT else : NEW_LINE
a [ arr [ i ] ] = i NEW_LINE if ( k == 0 ) : NEW_LINE print ( " No ▁ repeating ▁ element ▁ found " ) NEW_LINE else : NEW_LINE min = max + 1 NEW_LINE for i in range ( max + 1 ) : NEW_LINE
if ( a [ i ] and ( min > ( a [ i ] ) ) and b [ i ] ) : NEW_LINE INDENT min = a [ i ] NEW_LINE DEDENT print ( arr [ min ] ) NEW_LINE
arr = [ 10 , 5 , 3 , 4 , 3 , 5 , 6 ] NEW_LINE n = len ( arr ) NEW_LINE printFirstRepeating ( arr , n ) NEW_LINE
def printKDistinct ( arr , n , k ) : NEW_LINE INDENT dist_count = 0 NEW_LINE for i in range ( n ) : NEW_LINE DEDENT
j = 0 NEW_LINE while j < n : NEW_LINE INDENT if ( i != j and arr [ j ] == arr [ i ] ) : NEW_LINE INDENT break NEW_LINE DEDENT j += 1 NEW_LINE DEDENT
if ( j == n ) : NEW_LINE INDENT dist_count += 1 NEW_LINE DEDENT if ( dist_count == k ) : NEW_LINE INDENT return arr [ i ] NEW_LINE DEDENT return - 1 NEW_LINE
ar = [ 1 , 2 , 1 , 3 , 4 , 2 ] NEW_LINE n = len ( ar ) NEW_LINE k = 2 NEW_LINE print ( printKDistinct ( ar , n , k ) ) NEW_LINE
def countSubarrays ( A ) : NEW_LINE
res = 0 NEW_LINE
curr , cnt = A [ 0 ] , [ 1 ] NEW_LINE for c in A [ 1 : ] : NEW_LINE
if c == curr : NEW_LINE
cnt [ - 1 ] += 1 NEW_LINE else : NEW_LINE
curr = c NEW_LINE cnt . append ( 1 ) NEW_LINE
for i in range ( 1 , len ( cnt ) ) : NEW_LINE
res += min ( cnt [ i - 1 ] , cnt [ i ] ) NEW_LINE print ( res - 1 ) NEW_LINE
/ * Driver code * / NEW_LINE
A = [ 1 , 1 , 0 , 0 , 1 , 0 ] NEW_LINE
countSubarrays ( A ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . left = None NEW_LINE self . right = None NEW_LINE self . val = data NEW_LINE DEDENT DEDENT
def newNode ( data ) : NEW_LINE INDENT temp = Node ( data ) NEW_LINE return temp NEW_LINE DEDENT
def isEvenOddBinaryTree ( root ) : NEW_LINE INDENT if ( root == None ) : NEW_LINE INDENT return True NEW_LINE DEDENT q = [ ] NEW_LINE DEDENT
q . append ( root ) NEW_LINE
level = 0 NEW_LINE
while ( len ( q ) != 0 ) : NEW_LINE
size = len ( q ) NEW_LINE for i in range ( size ) : NEW_LINE INDENT node = q [ 0 ] NEW_LINE q . pop ( 0 ) NEW_LINE DEDENT
if ( level % 2 == 0 ) : NEW_LINE INDENT if ( node . val % 2 == 1 ) : NEW_LINE INDENT return False NEW_LINE DEDENT elif ( level % 2 == 1 ) : NEW_LINE INDENT if ( node . val % 2 == 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT DEDENT
if ( node . left != None ) : NEW_LINE INDENT q . append ( node . left ) NEW_LINE DEDENT if ( node . right != None ) : NEW_LINE INDENT q . append ( node . right ) NEW_LINE DEDENT
level += 1 NEW_LINE return True NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
root = None NEW_LINE root = newNode ( 2 ) NEW_LINE root . left = newNode ( 3 ) NEW_LINE root . right = newNode ( 9 ) NEW_LINE root . left . left = newNode ( 4 ) NEW_LINE root . left . right = newNode ( 10 ) NEW_LINE root . right . right = newNode ( 6 ) NEW_LINE
if ( isEvenOddBinaryTree ( root ) ) : NEW_LINE INDENT print ( " YES " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NO " ) NEW_LINE DEDENT
import sys NEW_LINE def findMaxLen ( a ) : NEW_LINE
freq = [ 0 ] * ( n + 1 ) NEW_LINE for i in range ( n ) : NEW_LINE INDENT freq [ a [ i ] ] += 1 NEW_LINE DEDENT maxFreqElement = - sys . maxsize - 1 NEW_LINE maxFreqCount = 1 NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE
if ( freq [ i ] > maxFreqElement ) : NEW_LINE INDENT maxFreqElement = freq [ i ] NEW_LINE maxFreqCount = 1 NEW_LINE DEDENT
elif ( freq [ i ] == maxFreqElement ) : NEW_LINE INDENT maxFreqCount += 1 NEW_LINE DEDENT
if ( maxFreqElement == 1 ) : NEW_LINE INDENT ans = 0 NEW_LINE DEDENT else : NEW_LINE
ans = ( ( n - maxFreqCount ) // ( maxFreqElement - 1 ) ) NEW_LINE
return ans NEW_LINE
a = [ 1 , 2 , 1 , 2 ] NEW_LINE print ( findMaxLen ( a ) ) NEW_LINE
import math NEW_LINE
def getMid ( s , e ) : NEW_LINE INDENT return ( s + ( e - s ) // 2 ) NEW_LINE DEDENT
def MaxUtil ( st , ss , se , l , r , node ) : NEW_LINE
if ( l <= ss and r >= se ) : NEW_LINE
return st [ node ] NEW_LINE
if ( se < l or ss > r ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
mid = getMid ( ss , se ) NEW_LINE return max ( MaxUtil ( st , ss , mid , l , r , 2 * node + 1 ) , MaxUtil ( st , mid + 1 , se , l , r , 2 * node + 2 ) ) NEW_LINE
def getMax ( st , n , l , r ) : NEW_LINE
if ( l < 0 or r > n - 1 or l > r ) : NEW_LINE INDENT print ( " Invalid ▁ Input " ) NEW_LINE return - 1 NEW_LINE DEDENT return MaxUtil ( st , 0 , n - 1 , l , r , 0 ) NEW_LINE
def constructSTUtil ( arr , ss , se , st , si ) : NEW_LINE
if ( ss == se ) : NEW_LINE INDENT st [ si ] = arr [ ss ] NEW_LINE return arr [ ss ] NEW_LINE DEDENT
mid = getMid ( ss , se ) NEW_LINE
def constructST ( arr , n ) : NEW_LINE
x = ( int ) ( math . ceil ( math . log ( n ) ) ) NEW_LINE
max_size = 2 * ( int ) ( pow ( 2 , x ) ) - 1 NEW_LINE
st = [ 0 ] * max_size NEW_LINE
constructSTUtil ( arr , 0 , n - 1 , st , 0 ) NEW_LINE
return st NEW_LINE
arr = [ 5 , 2 , 3 , 0 ] NEW_LINE n = len ( arr ) NEW_LINE
st = constructST ( arr , n ) NEW_LINE Q = [ [ 1 , 3 ] , [ 0 , 2 ] ] NEW_LINE for i in range ( len ( Q ) ) : NEW_LINE INDENT Max = getMax ( st , n , Q [ i ] [ 0 ] , Q [ i ] [ 1 ] ) NEW_LINE ok = 0 NEW_LINE for j in range ( 30 , - 1 , - 1 ) : NEW_LINE INDENT if ( ( Max & ( 1 << j ) ) != 0 ) : NEW_LINE INDENT ok = 1 NEW_LINE DEDENT if ( ok <= 0 ) : NEW_LINE INDENT continue NEW_LINE DEDENT Max |= ( 1 << j ) NEW_LINE DEDENT print ( Max , end = " ▁ " ) NEW_LINE DEDENT
def calculate ( a ) : NEW_LINE
a . sort ( ) NEW_LINE count = 1 NEW_LINE answer = 0 NEW_LINE
for i in range ( 1 , len ( a ) ) : NEW_LINE INDENT if a [ i ] == a [ i - 1 ] : NEW_LINE DEDENT
count += 1 NEW_LINE else : NEW_LINE
answer = answer + count * ( count - 1 ) // 2 NEW_LINE count = 1 NEW_LINE answer = answer + count * ( count - 1 ) // 2 NEW_LINE return answer NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT a = [ 1 , 2 , 1 , 2 , 4 ] NEW_LINE DEDENT
print ( calculate ( a ) ) NEW_LINE
def calculate ( a ) : NEW_LINE
maximum = max ( a ) NEW_LINE
frequency = [ 0 for x in range ( maximum + 1 ) ] NEW_LINE
for i in a : NEW_LINE
frequency [ i ] += 1 NEW_LINE answer = 0 NEW_LINE
for i in frequency : NEW_LINE
answer = answer + i * ( i - 1 ) // 2 NEW_LINE return answer NEW_LINE
a = [ 1 , 2 , 1 , 2 , 4 ] NEW_LINE
print ( calculate ( a ) ) NEW_LINE
def findSubArray ( arr , n ) : NEW_LINE INDENT sum = 0 NEW_LINE maxsize = - 1 NEW_LINE DEDENT
for i in range ( 0 , n - 1 ) : NEW_LINE INDENT sum = - 1 if ( arr [ i ] == 0 ) else 1 NEW_LINE DEDENT
for j in range ( i + 1 , n ) : NEW_LINE INDENT sum = sum + ( - 1 ) if ( arr [ j ] == 0 ) else sum + 1 NEW_LINE DEDENT
if ( sum == 0 and maxsize < j - i + 1 ) : NEW_LINE INDENT maxsize = j - i + 1 NEW_LINE startindex = i NEW_LINE DEDENT if ( maxsize == - 1 ) : NEW_LINE print ( " No ▁ such ▁ subarray " ) ; NEW_LINE else : NEW_LINE print ( startindex , " to " , startindex + maxsize - 1 ) ; NEW_LINE return maxsize NEW_LINE
arr = [ 1 , 0 , 0 , 1 , 0 , 1 , 1 ] NEW_LINE size = len ( arr ) NEW_LINE findSubArray ( arr , size ) NEW_LINE
def findMax ( arr , low , high ) : NEW_LINE
if ( high == low ) : NEW_LINE INDENT return arr [ low ] NEW_LINE DEDENT
mid = low + ( high - low ) // 2 NEW_LINE
if ( mid == 0 and arr [ mid ] > arr [ mid + 1 ] ) : NEW_LINE INDENT return arr [ mid ] NEW_LINE DEDENT
if ( arr [ low ] > arr [ mid ] ) : NEW_LINE INDENT return findMax ( arr , low , mid - 1 ) NEW_LINE DEDENT else : NEW_LINE INDENT return findMax ( arr , mid + 1 , high ) NEW_LINE DEDENT
arr = [ 6 , 5 , 4 , 3 , 2 , 1 ] NEW_LINE n = len ( arr ) NEW_LINE print ( findMax ( arr , 0 , n - 1 ) ) NEW_LINE
def ternarySearch ( l , r , key , ar ) : NEW_LINE INDENT while r >= l : NEW_LINE DEDENT
mid1 = l + ( r - l ) // 3 NEW_LINE mid2 = r - ( r - l ) // 3 NEW_LINE
if key == ar [ mid1 ] : NEW_LINE INDENT return mid1 NEW_LINE DEDENT if key == mid2 : NEW_LINE INDENT return mid2 NEW_LINE DEDENT
if key < ar [ mid1 ] : NEW_LINE
r = mid1 - 1 NEW_LINE elif key > ar [ mid2 ] : NEW_LINE
l = mid2 + 1 NEW_LINE else : NEW_LINE
l = mid1 + 1 NEW_LINE r = mid2 - 1 NEW_LINE
return - 1 NEW_LINE
ar = [ 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 ] NEW_LINE
l = 0 NEW_LINE
r = 9 NEW_LINE
key = 5 NEW_LINE
p = ternarySearch ( l , r , key , ar ) NEW_LINE
print ( " Index ▁ of " , key , " is " , p ) NEW_LINE
key = 50 NEW_LINE
p = ternarySearch ( l , r , key , ar ) NEW_LINE
print ( " Index ▁ of " , key , " is " , p ) NEW_LINE
def majorityNumber ( nums ) : NEW_LINE INDENT num_count = { } NEW_LINE for num in nums : NEW_LINE INDENT if num in num_count : NEW_LINE INDENT num_count [ num ] += 1 NEW_LINE DEDENT else : NEW_LINE INDENT num_count [ num ] = 1 NEW_LINE DEDENT DEDENT for num in num_count : NEW_LINE INDENT if num_count [ num ] > len ( nums ) / 2 : NEW_LINE INDENT return num NEW_LINE DEDENT DEDENT return - 1 NEW_LINE DEDENT
a = [ 2 , 2 , 1 , 1 , 1 , 2 , 2 ] NEW_LINE print majorityNumber ( a ) NEW_LINE
def search ( arr , l , h , key ) : NEW_LINE INDENT if l > h : NEW_LINE INDENT return - 1 NEW_LINE DEDENT mid = ( l + h ) // 2 NEW_LINE if arr [ mid ] == key : NEW_LINE INDENT return mid NEW_LINE DEDENT DEDENT
if arr [ l ] <= arr [ mid ] : NEW_LINE
if key >= arr [ l ] and key <= arr [ mid ] : NEW_LINE INDENT return search ( arr , l , mid - 1 , key ) NEW_LINE DEDENT
return search ( arr , mid + 1 , h , key ) NEW_LINE
if key >= arr [ mid ] and key <= arr [ h ] : NEW_LINE INDENT return search ( a , mid + 1 , h , key ) NEW_LINE DEDENT return search ( arr , l , mid - 1 , key ) NEW_LINE
arr = [ 4 , 5 , 6 , 7 , 8 , 9 , 1 , 2 , 3 ] NEW_LINE key = 6 NEW_LINE i = search ( arr , 0 , len ( arr ) - 1 , key ) NEW_LINE if i != - 1 : NEW_LINE INDENT print ( " Index : ▁ % ▁ d " % i ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Key ▁ not ▁ found " ) NEW_LINE DEDENT
def findMin ( arr , low , high ) : NEW_LINE
if high < low : NEW_LINE INDENT return arr [ 0 ] NEW_LINE DEDENT
if high == low : NEW_LINE INDENT return arr [ low ] NEW_LINE DEDENT
mid = int ( ( low + high ) / 2 ) NEW_LINE
if mid < high and arr [ mid + 1 ] < arr [ mid ] : NEW_LINE INDENT return arr [ mid + 1 ] NEW_LINE DEDENT
if mid > low and arr [ mid ] < arr [ mid - 1 ] : NEW_LINE INDENT return arr [ mid ] NEW_LINE DEDENT
if arr [ high ] > arr [ mid ] : NEW_LINE INDENT return findMin ( arr , low , mid - 1 ) NEW_LINE DEDENT return findMin ( arr , mid + 1 , high ) NEW_LINE
arr1 = [ 5 , 6 , 1 , 2 , 3 , 4 ] NEW_LINE n1 = len ( arr1 ) NEW_LINE print ( " The ▁ minimum ▁ element ▁ is ▁ " + str ( findMin ( arr1 , 0 , n1 - 1 ) ) ) NEW_LINE arr2 = [ 1 , 2 , 3 , 4 ] NEW_LINE n2 = len ( arr2 ) NEW_LINE print ( " The ▁ minimum ▁ element ▁ is ▁ " + str ( findMin ( arr2 , 0 , n2 - 1 ) ) ) NEW_LINE arr3 = [ 1 ] NEW_LINE n3 = len ( arr3 ) NEW_LINE print ( " The ▁ minimum ▁ element ▁ is ▁ " + str ( findMin ( arr3 , 0 , n3 - 1 ) ) ) NEW_LINE arr4 = [ 1 , 2 ] NEW_LINE n4 = len ( arr4 ) NEW_LINE print ( " The ▁ minimum ▁ element ▁ is ▁ " + str ( findMin ( arr4 , 0 , n4 - 1 ) ) ) NEW_LINE arr5 = [ 2 , 1 ] NEW_LINE n5 = len ( arr5 ) NEW_LINE print ( " The ▁ minimum ▁ element ▁ is ▁ " + str ( findMin ( arr5 , 0 , n5 - 1 ) ) ) NEW_LINE arr6 = [ 5 , 6 , 7 , 1 , 2 , 3 , 4 ] NEW_LINE n6 = len ( arr6 ) NEW_LINE print ( " The ▁ minimum ▁ element ▁ is ▁ " + str ( findMin ( arr6 , 0 , n6 - 1 ) ) ) NEW_LINE arr7 = [ 1 , 2 , 3 , 4 , 5 , 6 , 7 ] NEW_LINE n7 = len ( arr7 ) NEW_LINE print ( " The ▁ minimum ▁ element ▁ is ▁ " + str ( findMin ( arr7 , 0 , n7 - 1 ) ) ) NEW_LINE arr8 = [ 2 , 3 , 4 , 5 , 6 , 7 , 8 , 1 ] NEW_LINE n8 = len ( arr8 ) NEW_LINE print ( " The ▁ minimum ▁ element ▁ is ▁ " + str ( findMin ( arr8 , 0 , n8 - 1 ) ) ) NEW_LINE arr9 = [ 3 , 4 , 5 , 1 , 2 ] NEW_LINE n9 = len ( arr9 ) NEW_LINE print ( " The ▁ minimum ▁ element ▁ is ▁ " + str ( findMin ( arr9 , 0 , n9 - 1 ) ) ) NEW_LINE
def findMin ( arr , low , high ) : NEW_LINE INDENT while ( low < high ) : NEW_LINE INDENT mid = low + ( high - low ) // 2 ; NEW_LINE if ( arr [ mid ] == arr [ high ] ) : NEW_LINE INDENT high -= 1 ; NEW_LINE DEDENT elif ( arr [ mid ] > arr [ high ] ) : NEW_LINE INDENT low = mid + 1 ; NEW_LINE DEDENT else : NEW_LINE INDENT high = mid ; NEW_LINE DEDENT DEDENT return arr [ high ] ; NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr1 = [ 5 , 6 , 1 , 2 , 3 , 4 ] ; NEW_LINE n1 = len ( arr1 ) ; NEW_LINE print ( " The ▁ minimum ▁ element ▁ is ▁ " , findMin ( arr1 , 0 , n1 - 1 ) ) ; NEW_LINE arr2 = [ 1 , 2 , 3 , 4 ] ; NEW_LINE n2 = len ( arr2 ) ; NEW_LINE print ( " The ▁ minimum ▁ element ▁ is ▁ " , findMin ( arr2 , 0 , n2 - 1 ) ) ; NEW_LINE arr3 = [ 1 ] ; NEW_LINE n3 = len ( arr3 ) ; NEW_LINE print ( " The ▁ minimum ▁ element ▁ is ▁ " , findMin ( arr3 , 0 , n3 - 1 ) ) ; NEW_LINE arr4 = [ 1 , 2 ] ; NEW_LINE n4 = len ( arr4 ) ; NEW_LINE print ( " The ▁ minimum ▁ element ▁ is ▁ " , findMin ( arr4 , 0 , n4 - 1 ) ) ; NEW_LINE arr5 = [ 2 , 1 ] ; NEW_LINE n5 = len ( arr5 ) ; NEW_LINE print ( " The ▁ minimum ▁ element ▁ is ▁ " , findMin ( arr5 , 0 , n5 - 1 ) ) ; NEW_LINE arr6 = [ 5 , 6 , 7 , 1 , 2 , 3 , 4 ] ; NEW_LINE n6 = len ( arr6 ) ; NEW_LINE print ( " The ▁ minimum ▁ element ▁ is ▁ " , findMin ( arr6 , 0 , n6 - 1 ) ) ; NEW_LINE arr7 = [ 1 , 2 , 3 , 4 , 5 , 6 , 7 ] ; NEW_LINE n7 = len ( arr7 ) ; NEW_LINE print ( " The ▁ minimum ▁ element ▁ is ▁ " , findMin ( arr7 , 0 , n7 - 1 ) ) ; NEW_LINE arr8 = [ 2 , 3 , 4 , 5 , 6 , 7 , 8 , 1 ] ; NEW_LINE n8 = len ( arr8 ) ; NEW_LINE print ( " The ▁ minimum ▁ element ▁ is ▁ " , findMin ( arr8 , 0 , n8 - 1 ) ) ; NEW_LINE arr9 = [ 3 , 4 , 5 , 1 , 2 ] ; NEW_LINE n9 = len ( arr9 ) ; NEW_LINE print ( " The ▁ minimum ▁ element ▁ is ▁ " , findMin ( arr9 , 0 , n9 - 1 ) ) ; NEW_LINE DEDENT
from bisect import bisect as upper_bound NEW_LINE
def countPairs ( a , n , mid ) : NEW_LINE INDENT res = 0 NEW_LINE for i in range ( n ) : NEW_LINE DEDENT
res += upper_bound ( a , a [ i ] + mid ) NEW_LINE return res NEW_LINE
def kthDiff ( a , n , k ) : NEW_LINE
a = sorted ( a ) NEW_LINE
low = a [ 1 ] - a [ 0 ] NEW_LINE for i in range ( 1 , n - 1 ) : NEW_LINE INDENT low = min ( low , a [ i + 1 ] - a [ i ] ) NEW_LINE DEDENT
high = a [ n - 1 ] - a [ 0 ] NEW_LINE
while ( low < high ) : NEW_LINE INDENT mid = ( low + high ) >> 1 NEW_LINE if ( countPairs ( a , n , mid ) < k ) : NEW_LINE INDENT low = mid + 1 NEW_LINE DEDENT else : NEW_LINE INDENT high = mid NEW_LINE DEDENT DEDENT return low NEW_LINE
k = 3 NEW_LINE a = [ 1 , 2 , 3 , 4 ] NEW_LINE n = len ( a ) NEW_LINE print ( kthDiff ( a , n , k ) ) NEW_LINE
import sys NEW_LINE
def print2Smallest ( arr ) : NEW_LINE
arr_size = len ( arr ) NEW_LINE if arr_size < 2 : NEW_LINE INDENT print " Invalid ▁ Input " NEW_LINE return NEW_LINE DEDENT first = second = sys . maxint NEW_LINE for i in range ( 0 , arr_size ) : NEW_LINE
if arr [ i ] < first : NEW_LINE INDENT second = first NEW_LINE first = arr [ i ] NEW_LINE DEDENT
elif ( arr [ i ] < second and arr [ i ] != first ) : NEW_LINE INDENT second = arr [ i ] ; NEW_LINE DEDENT if ( second == sys . maxint ) : NEW_LINE print " No ▁ second ▁ smallest ▁ element " NEW_LINE else : NEW_LINE print ' The ▁ smallest ▁ element ▁ is ' , first , ' and ' ' ▁ second ▁ smallest ▁ element ▁ is ' , second NEW_LINE
arr = [ 12 , 13 , 1 , 10 , 34 , 1 ] NEW_LINE print2Smallest ( arr ) NEW_LINE
MAX = 1000 NEW_LINE
tree = [ 0 ] * ( 4 * MAX ) NEW_LINE
arr = [ 0 ] * MAX NEW_LINE
def gcd ( a : int , b : int ) : NEW_LINE INDENT if a == 0 : NEW_LINE INDENT return b NEW_LINE DEDENT return gcd ( b % a , a ) NEW_LINE DEDENT
def lcm ( a : int , b : int ) : NEW_LINE INDENT return ( a * b ) // gcd ( a , b ) NEW_LINE DEDENT
def build ( node : int , start : int , end : int ) : NEW_LINE
if start == end : NEW_LINE INDENT tree [ node ] = arr [ start ] NEW_LINE return NEW_LINE DEDENT mid = ( start + end ) // 2 NEW_LINE
build ( 2 * node , start , mid ) NEW_LINE build ( 2 * node + 1 , mid + 1 , end ) NEW_LINE
left_lcm = tree [ 2 * node ] NEW_LINE right_lcm = tree [ 2 * node + 1 ] NEW_LINE tree [ node ] = lcm ( left_lcm , right_lcm ) NEW_LINE
def query ( node : int , start : int , end : int , l : int , r : int ) : NEW_LINE
if end < l or start > r : NEW_LINE INDENT return 1 NEW_LINE DEDENT
if l <= start and r >= end : NEW_LINE INDENT return tree [ node ] NEW_LINE DEDENT
mid = ( start + end ) // 2 NEW_LINE left_lcm = query ( 2 * node , start , mid , l , r ) NEW_LINE right_lcm = query ( 2 * node + 1 , mid + 1 , end , l , r ) NEW_LINE return lcm ( left_lcm , right_lcm ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
arr [ 0 ] = 5 NEW_LINE arr [ 1 ] = 7 NEW_LINE arr [ 2 ] = 5 NEW_LINE arr [ 3 ] = 2 NEW_LINE arr [ 4 ] = 10 NEW_LINE arr [ 5 ] = 12 NEW_LINE arr [ 6 ] = 11 NEW_LINE arr [ 7 ] = 17 NEW_LINE arr [ 8 ] = 14 NEW_LINE arr [ 9 ] = 1 NEW_LINE arr [ 10 ] = 44 NEW_LINE
build ( 1 , 0 , 10 ) NEW_LINE
print ( query ( 1 , 0 , 10 , 2 , 5 ) ) NEW_LINE
print ( query ( 1 , 0 , 10 , 5 , 10 ) ) NEW_LINE
print ( query ( 1 , 0 , 10 , 0 , 10 ) ) NEW_LINE
M = 1000000007 NEW_LINE def waysOfDecoding ( s ) : NEW_LINE INDENT dp = [ 0 ] * ( len ( s ) + 1 ) NEW_LINE dp [ 0 ] = 1 NEW_LINE DEDENT
if s [ 0 ] == ' * ' : NEW_LINE INDENT dp [ 1 ] = 9 NEW_LINE DEDENT elif s [ 0 ] == '0' : NEW_LINE INDENT dp [ 1 ] = 0 NEW_LINE DEDENT else : NEW_LINE INDENT dp [ 1 ] = 1 NEW_LINE DEDENT
for i in range ( len ( s ) ) : NEW_LINE
if ( s [ i ] == ' * ' ) : NEW_LINE INDENT dp [ i + 1 ] = 9 * dp [ i ] NEW_LINE DEDENT
if ( s [ i - 1 ] == '1' ) : NEW_LINE INDENT dp [ i + 1 ] = ( dp [ i + 1 ] + 9 * dp [ i - 1 ] ) % M NEW_LINE DEDENT
elif ( s [ i - 1 ] == '2' ) : NEW_LINE INDENT dp [ i + 1 ] = ( dp [ i + 1 ] + 6 * dp [ i - 1 ] ) % M NEW_LINE DEDENT
elif ( s [ i - 1 ] == ' * ' ) : NEW_LINE INDENT dp [ i + 1 ] = ( dp [ i + 1 ] + 15 * dp [ i - 1 ] ) % M NEW_LINE DEDENT else : NEW_LINE
if s [ i ] != '0' : NEW_LINE INDENT dp [ i + 1 ] = dp [ i ] NEW_LINE DEDENT else : NEW_LINE INDENT dp [ i + 1 ] = 0 NEW_LINE DEDENT
if ( s [ i - 1 ] == '1' ) : NEW_LINE INDENT dp [ i + 1 ] = ( dp [ i + 1 ] + dp [ i - 1 ] ) % M NEW_LINE DEDENT
elif ( s [ i - 1 ] == '2' and s [ i ] <= '6' ) : NEW_LINE INDENT dp [ i + 1 ] = ( dp [ i + 1 ] + dp [ i - 1 ] ) % M NEW_LINE DEDENT
elif ( s [ i - 1 ] == ' * ' ) : NEW_LINE INDENT if ( s [ i ] <= '6' ) : NEW_LINE INDENT dp [ i + 1 ] = dp [ i + 1 ] + 2 * dp [ i - 1 ] NEW_LINE DEDENT else : NEW_LINE INDENT dp [ i + 1 ] = dp [ i + 1 ] + 1 * dp [ i - 1 ] NEW_LINE DEDENT dp [ i + 1 ] = dp [ i + 1 ] % M NEW_LINE DEDENT return dp [ len ( s ) ] NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT s = "12" NEW_LINE print ( waysOfDecoding ( s ) ) NEW_LINE DEDENT
def countSubset ( arr , n , diff ) : NEW_LINE
sum = 0 NEW_LINE for i in range ( n ) : NEW_LINE INDENT sum += arr [ i ] NEW_LINE DEDENT sum += diff NEW_LINE sum = sum // 2 NEW_LINE
t = [ [ 0 for i in range ( sum + 1 ) ] for i in range ( n + 1 ) ] NEW_LINE
for j in range ( sum + 1 ) : NEW_LINE INDENT t [ 0 ] [ j ] = 0 NEW_LINE DEDENT
for i in range ( n + 1 ) : NEW_LINE INDENT t [ i ] [ 0 ] = 1 NEW_LINE DEDENT
for i in range ( 1 , n + 1 ) : NEW_LINE INDENT for j in range ( 1 , sum + 1 ) : NEW_LINE DEDENT
if ( arr [ i - 1 ] > j ) : NEW_LINE INDENT t [ i ] [ j ] = t [ i - 1 ] [ j ] NEW_LINE DEDENT else : NEW_LINE INDENT t [ i ] [ j ] = t [ i - 1 ] [ j ] + t [ i - 1 ] [ j - arr [ i - 1 ] ] NEW_LINE DEDENT
return t [ n ] [ sum ] NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
diff , n = 1 , 4 NEW_LINE arr = [ 1 , 1 , 2 , 3 ] NEW_LINE
print ( countSubset ( arr , n , diff ) ) NEW_LINE
dp = [ [ 0 for i in range ( 605 ) ] for j in range ( 105 ) ] NEW_LINE
def find ( N , a , b ) : NEW_LINE INDENT probability = 0.0 NEW_LINE DEDENT
for i in range ( 1 , 7 ) : NEW_LINE INDENT dp [ 1 ] [ i ] = 1.0 / 6 NEW_LINE DEDENT for i in range ( 2 , N + 1 ) : NEW_LINE INDENT for j in range ( i , ( 6 * i ) + 1 ) : NEW_LINE INDENT for k in range ( 1 , 7 ) : NEW_LINE INDENT dp [ i ] [ j ] = dp [ i ] [ j ] + dp [ i - 1 ] [ j - k ] / 6 NEW_LINE DEDENT DEDENT DEDENT
for Sum in range ( a , b + 1 ) : NEW_LINE INDENT probability = probability + dp [ N ] [ Sum ] NEW_LINE DEDENT return probability NEW_LINE
N , a , b = 4 , 13 , 17 NEW_LINE probability = find ( N , a , b ) NEW_LINE
print ( ' % .6f ' % probability ) NEW_LINE
from collections import deque as queue NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , x ) : NEW_LINE INDENT self . data = x NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def getSumAlternate ( root ) : NEW_LINE INDENT if ( root == None ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT sum = root . data NEW_LINE if ( root . left != None ) : NEW_LINE INDENT sum += getSum ( root . left . left ) NEW_LINE sum += getSum ( root . left . right ) NEW_LINE DEDENT if ( root . right != None ) : NEW_LINE INDENT sum += getSum ( root . right . left ) NEW_LINE sum += getSum ( root . right . right ) NEW_LINE DEDENT return sum NEW_LINE DEDENT
def getSum ( root ) : NEW_LINE INDENT if ( root == None ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT DEDENT
return max ( getSumAlternate ( root ) , ( getSumAlternate ( root . left ) + getSumAlternate ( root . right ) ) ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT root = Node ( 1 ) NEW_LINE root . left = Node ( 2 ) NEW_LINE root . right = Node ( 3 ) NEW_LINE root . right . left = Node ( 4 ) NEW_LINE root . right . left . right = Node ( 5 ) NEW_LINE root . right . left . right . left = Node ( 6 ) NEW_LINE print ( getSum ( root ) ) NEW_LINE DEDENT
def isSubsetSum ( arr , n , sum ) : NEW_LINE
subset = [ [ False for j in range ( sum + 1 ) ] for i in range ( 3 ) ] NEW_LINE for i in range ( n + 1 ) : NEW_LINE INDENT for j in range ( sum + 1 ) : NEW_LINE DEDENT
if ( j == 0 ) : NEW_LINE INDENT subset [ i % 2 ] [ j ] = True NEW_LINE DEDENT
elif ( i == 0 ) : NEW_LINE INDENT subset [ i % 2 ] [ j ] = False NEW_LINE DEDENT elif ( arr [ i - 1 ] <= j ) : NEW_LINE INDENT subset [ i % 2 ] [ j ] = subset [ ( i + 1 ) % 2 ] [ j - arr [ i - 1 ] ] or subset [ ( i + 1 ) % 2 ] [ j ] NEW_LINE DEDENT else : NEW_LINE INDENT subset [ i % 2 ] [ j ] = subset [ ( i + 1 ) % 2 ] [ j ] NEW_LINE DEDENT return subset [ n % 2 ] [ sum ] NEW_LINE
arr = [ 6 , 2 , 5 ] NEW_LINE sum = 7 NEW_LINE n = len ( arr ) NEW_LINE if ( isSubsetSum ( arr , n , sum ) == True ) : NEW_LINE INDENT print ( " There ▁ exists ▁ a ▁ subset ▁ with ▁ given ▁ sum " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No ▁ subset ▁ exists ▁ with ▁ given ▁ sum " ) NEW_LINE DEDENT
import sys NEW_LINE
def findMaxSum ( arr , n ) : NEW_LINE INDENT res = - sys . maxsize - 1 NEW_LINE for i in range ( n ) : NEW_LINE INDENT prefix_sum = arr [ i ] NEW_LINE for j in range ( i ) : NEW_LINE INDENT prefix_sum += arr [ j ] NEW_LINE DEDENT suffix_sum = arr [ i ] NEW_LINE j = n - 1 NEW_LINE while ( j > i ) : NEW_LINE INDENT suffix_sum += arr [ j ] NEW_LINE j -= 1 NEW_LINE DEDENT if ( prefix_sum == suffix_sum ) : NEW_LINE INDENT res = max ( res , prefix_sum ) NEW_LINE DEDENT DEDENT return res NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ - 2 , 5 , 3 , 1 , 2 , 6 , - 4 , 2 ] NEW_LINE n = len ( arr ) NEW_LINE print ( findMaxSum ( arr , n ) ) NEW_LINE DEDENT
def findMaxSum ( arr , n ) : NEW_LINE
preSum = [ 0 for i in range ( n ) ] NEW_LINE
suffSum = [ 0 for i in range ( n ) ] NEW_LINE
ans = - 10000000 NEW_LINE
preSum [ 0 ] = arr [ 0 ] NEW_LINE for i in range ( 1 , n ) : NEW_LINE INDENT preSum [ i ] = preSum [ i - 1 ] + arr [ i ] NEW_LINE DEDENT
suffSum [ n - 1 ] = arr [ n - 1 ] NEW_LINE if ( preSum [ n - 1 ] == suffSum [ n - 1 ] ) : NEW_LINE INDENT ans = max ( ans , preSum [ n - 1 ] ) NEW_LINE DEDENT for i in range ( n - 2 , - 1 , - 1 ) : NEW_LINE INDENT suffSum [ i ] = suffSum [ i + 1 ] + arr [ i ] NEW_LINE if ( suffSum [ i ] == preSum [ i ] ) : NEW_LINE INDENT ans = max ( ans , preSum [ i ] ) NEW_LINE DEDENT DEDENT return ans NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ - 2 , 5 , 3 , 1 , 2 , 6 , - 4 , 2 ] NEW_LINE n = len ( arr ) NEW_LINE print ( findMaxSum ( arr , n ) ) NEW_LINE DEDENT
import sys NEW_LINE
def findMaxSum ( arr , n ) : NEW_LINE INDENT ss = sum ( arr ) NEW_LINE prefix_sum = 0 NEW_LINE res = - sys . maxsize NEW_LINE for i in range ( n ) : NEW_LINE INDENT prefix_sum += arr [ i ] NEW_LINE if prefix_sum == ss : NEW_LINE INDENT res = max ( res , prefix_sum ) ; NEW_LINE DEDENT ss -= arr [ i ] ; NEW_LINE DEDENT return res NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ - 2 , 5 , 3 , 1 , 2 , 6 , - 4 , 2 ] NEW_LINE n = len ( arr ) NEW_LINE print ( findMaxSum ( arr , n ) ) NEW_LINE DEDENT
def findMajority ( arr , n ) : NEW_LINE INDENT maxCount = 0 NEW_LINE DEDENT
index = - 1 NEW_LINE for i in range ( n ) : NEW_LINE INDENT count = 0 NEW_LINE for j in range ( n ) : NEW_LINE INDENT if ( arr [ i ] == arr [ j ] ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT DEDENT DEDENT
if ( count > maxCount ) : NEW_LINE INDENT maxCount = count NEW_LINE index = i NEW_LINE DEDENT
if ( maxCount > n // 2 ) : NEW_LINE INDENT print ( arr [ index ] ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No ▁ Majority ▁ Element " ) NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 1 , 1 , 2 , 1 , 3 , 5 , 1 ] NEW_LINE n = len ( arr ) NEW_LINE DEDENT
findMajority ( arr , n ) NEW_LINE
def findCandidate ( A ) : NEW_LINE INDENT maj_index = 0 NEW_LINE count = 1 NEW_LINE for i in range ( len ( A ) ) : NEW_LINE INDENT if A [ maj_index ] == A [ i ] : NEW_LINE INDENT count += 1 NEW_LINE DEDENT else : NEW_LINE INDENT count -= 1 NEW_LINE DEDENT if count == 0 : NEW_LINE INDENT maj_index = i NEW_LINE count = 1 NEW_LINE DEDENT DEDENT return A [ maj_index ] NEW_LINE DEDENT
def isMajority ( A , cand ) : NEW_LINE INDENT count = 0 NEW_LINE for i in range ( len ( A ) ) : NEW_LINE INDENT if A [ i ] == cand : NEW_LINE INDENT count += 1 NEW_LINE DEDENT DEDENT if count > len ( A ) / 2 : NEW_LINE INDENT return True NEW_LINE DEDENT else : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT
def printMajority ( A ) : NEW_LINE
cand = findCandidate ( A ) NEW_LINE
if isMajority ( A , cand ) == True : NEW_LINE INDENT print ( cand ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No ▁ Majority ▁ Element " ) NEW_LINE DEDENT
A = [ 1 , 3 , 3 , 1 , 2 ] NEW_LINE
printMajority ( A ) NEW_LINE
def findMajority ( arr , size ) : NEW_LINE INDENT m = { } NEW_LINE for i in range ( size ) : NEW_LINE INDENT if arr [ i ] in m : NEW_LINE INDENT m [ arr [ i ] ] += 1 NEW_LINE DEDENT else : NEW_LINE INDENT m [ arr [ i ] ] = 1 NEW_LINE DEDENT DEDENT count = 0 NEW_LINE for key in m : NEW_LINE INDENT if m [ key ] > size / 2 : NEW_LINE INDENT count = 1 NEW_LINE print ( " Majority ▁ found ▁ : - " , key ) NEW_LINE break NEW_LINE DEDENT DEDENT if ( count == 0 ) : NEW_LINE INDENT print ( " No ▁ Majority ▁ element " ) NEW_LINE DEDENT DEDENT
arr = [ 2 , 2 , 2 , 2 , 5 , 5 , 2 , 3 , 3 ] NEW_LINE n = len ( arr ) NEW_LINE
findMajority ( arr , n ) NEW_LINE
def majorityElement ( arr , n ) : NEW_LINE
arr . sort ( ) NEW_LINE count , max_ele , temp , f = 1 , - 1 , arr [ 0 ] , 0 NEW_LINE for i in range ( 1 , n ) : NEW_LINE
if ( temp == arr [ i ] ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT else : NEW_LINE INDENT count = 1 NEW_LINE temp = arr [ i ] NEW_LINE DEDENT
if ( max_ele < count ) : NEW_LINE INDENT max_ele = count NEW_LINE ele = arr [ i ] NEW_LINE if ( max_ele > ( n // 2 ) ) : NEW_LINE INDENT f = 1 NEW_LINE break NEW_LINE DEDENT DEDENT
if f == 1 : NEW_LINE INDENT return ele NEW_LINE DEDENT else : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
arr = [ 1 , 1 , 2 , 1 , 3 , 5 , 1 ] NEW_LINE n = len ( arr ) NEW_LINE
print ( majorityElement ( arr , n ) ) NEW_LINE
def isSubsetSum ( set , n , sum ) : NEW_LINE
subset = ( [ [ False for i in range ( sum + 1 ) ] for i in range ( n + 1 ) ] ) NEW_LINE
for i in range ( n + 1 ) : NEW_LINE INDENT subset [ i ] [ 0 ] = True NEW_LINE DEDENT
for i in range ( 1 , sum + 1 ) : NEW_LINE INDENT subset [ 0 ] [ i ] = False NEW_LINE DEDENT
for i in range ( 1 , n + 1 ) : NEW_LINE INDENT for j in range ( 1 , sum + 1 ) : NEW_LINE INDENT if j < set [ i - 1 ] : NEW_LINE INDENT subset [ i ] [ j ] = subset [ i - 1 ] [ j ] NEW_LINE DEDENT if j >= set [ i - 1 ] : NEW_LINE INDENT subset [ i ] [ j ] = ( subset [ i - 1 ] [ j ] or subset [ i - 1 ] [ j - set [ i - 1 ] ] ) NEW_LINE DEDENT DEDENT DEDENT
for i in range ( n + 1 ) : NEW_LINE for j in range ( sum + 1 ) : NEW_LINE print ( subset [ i ] [ j ] , end = " ▁ " ) NEW_LINE print ( ) NEW_LINE return subset [ n ] [ sum ] NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT set = [ 3 , 34 , 4 , 12 , 5 , 2 ] NEW_LINE sum = 9 NEW_LINE n = len ( set ) NEW_LINE if ( isSubsetSum ( set , n , sum ) == True ) : NEW_LINE INDENT print ( " Found ▁ a ▁ subset ▁ with ▁ given ▁ sum " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No ▁ subset ▁ with ▁ given ▁ sum " ) NEW_LINE DEDENT DEDENT
tab = [ [ - 1 for i in range ( 2000 ) ] for j in range ( 2000 ) ] NEW_LINE
def subsetSum ( a , n , sum ) : NEW_LINE
if ( sum == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT if ( n <= 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
if ( tab [ n - 1 ] [ sum ] != - 1 ) : NEW_LINE INDENT return tab [ n - 1 ] [ sum ] NEW_LINE DEDENT
if ( a [ n - 1 ] > sum ) : NEW_LINE INDENT tab [ n - 1 ] [ sum ] = subsetSum ( a , n - 1 , sum ) NEW_LINE return tab [ n - 1 ] [ sum ] NEW_LINE DEDENT else : NEW_LINE
tab [ n - 1 ] [ sum ] = subsetSum ( a , n - 1 , sum ) NEW_LINE return tab [ n - 1 ] [ sum ] or subsetSum ( a , n - 1 , sum - a [ n - 1 ] ) NEW_LINE
n = 5 NEW_LINE a = [ 1 , 5 , 3 , 7 , 4 ] NEW_LINE sum = 12 NEW_LINE if ( subsetSum ( a , n , sum ) ) : NEW_LINE INDENT print ( " YES " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NO " ) NEW_LINE DEDENT
from math import log NEW_LINE
def binpow ( a , b ) : NEW_LINE INDENT res = 1 NEW_LINE while ( b > 0 ) : NEW_LINE INDENT if ( b % 2 == 1 ) : NEW_LINE INDENT res = res * a NEW_LINE DEDENT a = a * a NEW_LINE b //= 2 NEW_LINE DEDENT return res NEW_LINE DEDENT
def find ( x ) : NEW_LINE INDENT if ( x == 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT p = log ( x ) / log ( 2 ) NEW_LINE return binpow ( 2 , p + 1 ) - 1 NEW_LINE DEDENT
def getBinary ( n ) : NEW_LINE
ans = " " NEW_LINE
while ( n > 0 ) : NEW_LINE INDENT dig = n % 2 NEW_LINE ans += str ( dig ) NEW_LINE n //= 2 NEW_LINE DEDENT
return ans NEW_LINE
def totalCountDifference ( n ) : NEW_LINE
ans = getBinary ( n ) NEW_LINE
req = 0 NEW_LINE
for i in range ( len ( ans ) ) : NEW_LINE
if ( ans [ i ] == '1' ) : NEW_LINE INDENT req += find ( binpow ( 2 , i ) ) NEW_LINE DEDENT return req NEW_LINE
N = 5 NEW_LINE
print ( totalCountDifference ( N ) ) NEW_LINE
def Maximum_Length ( a ) : NEW_LINE
counts = [ 0 ] * 11 NEW_LINE
for index , v in enumerate ( a ) : NEW_LINE
counts [ v ] += 1 NEW_LINE
k = sorted ( [ i for i in counts if i ] ) NEW_LINE
if len ( k ) == 1 or ( k [ 0 ] == k [ - 2 ] and k [ - 1 ] - k [ - 2 ] == 1 ) or ( k [ 0 ] == 1 and k [ 1 ] == k [ - 1 ] ) : NEW_LINE INDENT ans = index NEW_LINE DEDENT
return ans + 1 NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = [ 1 , 1 , 1 , 2 , 2 , 2 ] NEW_LINE n = len ( a ) NEW_LINE print ( Maximum_Length ( a ) ) NEW_LINE DEDENT
def gcd ( a , b ) : NEW_LINE INDENT if a == 0 : NEW_LINE INDENT return b NEW_LINE DEDENT return gcd ( b % a , a ) NEW_LINE DEDENT
def print_gcd_online ( n , m , query , arr ) : NEW_LINE
max_gcd = 0 NEW_LINE
for i in range ( 0 , n ) : NEW_LINE INDENT max_gcd = gcd ( max_gcd , arr [ i ] ) NEW_LINE DEDENT
for i in range ( 0 , m ) : NEW_LINE
query [ i ] [ 0 ] -= 1 NEW_LINE
arr [ query [ i ] [ 0 ] ] //= query [ i ] [ 1 ] NEW_LINE
max_gcd = gcd ( arr [ query [ i ] [ 0 ] ] , max_gcd ) NEW_LINE
print ( max_gcd ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n , m = 3 , 3 NEW_LINE query = [ [ 1 , 3 ] , [ 3 , 12 ] , [ 2 , 4 ] ] NEW_LINE arr = [ 36 , 24 , 72 ] NEW_LINE print_gcd_online ( n , m , query , arr ) NEW_LINE DEDENT
MAX = 1000000 NEW_LINE
prime = [ True ] * ( MAX + 1 ) NEW_LINE
sum = [ 0 ] * ( MAX + 1 ) NEW_LINE
def SieveOfEratosthenes ( ) : NEW_LINE
prime [ 1 ] = False NEW_LINE p = 2 NEW_LINE while p * p <= MAX : NEW_LINE
if ( prime [ p ] ) : NEW_LINE
i = p * 2 NEW_LINE while i <= MAX : NEW_LINE INDENT prime [ i ] = False NEW_LINE i += p NEW_LINE DEDENT p += 1 NEW_LINE
' NEW_LINE INDENT for i in range ( 1 , MAX + 1 ) : NEW_LINE INDENT if ( prime [ i ] == True ) : NEW_LINE INDENT sum [ i ] = 1 NEW_LINE DEDENT sum [ i ] += sum [ i - 1 ] NEW_LINE DEDENT DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE
SieveOfEratosthenes ( ) NEW_LINE
l = 3 NEW_LINE r = 9 NEW_LINE
c = ( sum [ r ] - sum [ l - 1 ] ) NEW_LINE
print ( " Count : " , c ) NEW_LINE
from math import pow , sqrt NEW_LINE
def area ( r ) : NEW_LINE
if ( r < 0 ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
area = 3.14 * pow ( r / ( 2 * sqrt ( 2 ) ) , 2 ) ; NEW_LINE return area ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT a = 5 NEW_LINE print ( " { 0 : . 6 } " . format ( area ( a ) ) ) NEW_LINE DEDENT
from math import * NEW_LINE N = 100005 NEW_LINE
prime = [ True ] * N NEW_LINE def SieveOfEratosthenes ( ) : NEW_LINE INDENT prime [ 1 ] = False NEW_LINE for p in range ( 2 , int ( sqrt ( N ) ) ) : NEW_LINE DEDENT
if prime [ p ] == True : NEW_LINE
for i in range ( 2 * p , N , p ) : NEW_LINE INDENT prime [ i ] = False NEW_LINE DEDENT
def almostPrimes ( n ) : NEW_LINE
ans = 0 NEW_LINE
for i in range ( 6 , n + 1 ) : NEW_LINE
c = 0 NEW_LINE for j in range ( 2 , int ( sqrt ( i ) ) + 1 ) : NEW_LINE
if i % j == 0 : NEW_LINE INDENT if j * j == i : NEW_LINE INDENT if prime [ j ] : NEW_LINE INDENT c += 1 NEW_LINE DEDENT DEDENT else : NEW_LINE INDENT if prime [ j ] : NEW_LINE INDENT c += 1 NEW_LINE DEDENT if prime [ i // j ] : NEW_LINE INDENT c += 1 NEW_LINE DEDENT DEDENT DEDENT
if c == 2 : NEW_LINE INDENT ans += 1 NEW_LINE DEDENT return ans NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT SieveOfEratosthenes ( ) NEW_LINE n = 21 NEW_LINE print ( almostPrimes ( n ) ) NEW_LINE DEDENT
def sumOfDigitsSingle ( x ) : NEW_LINE INDENT ans = 0 NEW_LINE while x : NEW_LINE INDENT ans += x % 10 NEW_LINE x //= 10 NEW_LINE DEDENT return ans NEW_LINE DEDENT
def closest ( x ) : NEW_LINE INDENT ans = 0 NEW_LINE while ( ans * 10 + 9 <= x ) : NEW_LINE INDENT ans = ans * 10 + 9 NEW_LINE DEDENT return ans NEW_LINE DEDENT def sumOfDigitsTwoParts ( N ) : NEW_LINE INDENT A = closest ( N ) NEW_LINE return sumOfDigitsSingle ( A ) + sumOfDigitsSingle ( N - A ) NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT N = 35 NEW_LINE print ( sumOfDigitsTwoParts ( N ) ) NEW_LINE DEDENT
def isPrime ( p ) : NEW_LINE
checkNumber = 2 ** p - 1 NEW_LINE
nextval = 4 % checkNumber NEW_LINE
for i in range ( 1 , p - 1 ) : NEW_LINE INDENT nextval = ( nextval * nextval - 2 ) % checkNumber NEW_LINE DEDENT
if ( nextval == 0 ) : return True NEW_LINE else : return False NEW_LINE
p = 7 NEW_LINE checkNumber = 2 ** p - 1 NEW_LINE if isPrime ( p ) : NEW_LINE INDENT print ( checkNumber , ' is ▁ Prime . ' ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( checkNumber , ' is ▁ not ▁ Prime ' ) NEW_LINE DEDENT
def sieve ( n , prime ) : NEW_LINE INDENT p = 2 NEW_LINE while ( p * p <= n ) : NEW_LINE DEDENT
if ( prime [ p ] == True ) : NEW_LINE
for i in range ( p * 2 , n , p ) : NEW_LINE INDENT prime [ i ] = False NEW_LINE DEDENT p += 1 NEW_LINE def printSophieGermanNumber ( n ) : NEW_LINE
prime = [ True ] * ( 2 * n + 1 ) NEW_LINE sieve ( 2 * n + 1 , prime ) NEW_LINE for i in range ( 2 , n + 1 ) : NEW_LINE
if ( prime [ i ] and prime [ 2 * i + 1 ] ) : NEW_LINE INDENT print ( i , end = " ▁ " ) NEW_LINE DEDENT
n = 25 NEW_LINE printSophieGermanNumber ( n ) NEW_LINE
def ucal ( u , n ) : NEW_LINE INDENT if ( n == 0 ) : NEW_LINE INDENT return 1 ; NEW_LINE DEDENT temp = u ; NEW_LINE for i in range ( 1 , int ( n / 2 + 1 ) ) : NEW_LINE INDENT temp = temp * ( u - i ) ; NEW_LINE DEDENT for i in range ( 1 , int ( n / 2 ) ) : NEW_LINE INDENT temp = temp * ( u + i ) ; NEW_LINE DEDENT return temp ; NEW_LINE DEDENT
def fact ( n ) : NEW_LINE INDENT f = 1 ; NEW_LINE for i in range ( 2 , n + 1 ) : NEW_LINE INDENT f *= i ; NEW_LINE DEDENT return f ; NEW_LINE DEDENT
n = 6 ; NEW_LINE x = [ 25 , 26 , 27 , 28 , 29 , 30 ] ; NEW_LINE
y = [ [ 0 for i in range ( n ) ] for j in range ( n ) ] ; NEW_LINE y [ 0 ] [ 0 ] = 4.000 ; NEW_LINE y [ 1 ] [ 0 ] = 3.846 ; NEW_LINE y [ 2 ] [ 0 ] = 3.704 ; NEW_LINE y [ 3 ] [ 0 ] = 3.571 ; NEW_LINE y [ 4 ] [ 0 ] = 3.448 ; NEW_LINE y [ 5 ] [ 0 ] = 3.333 ; NEW_LINE
for i in range ( 1 , n ) : NEW_LINE INDENT for j in range ( n - i ) : NEW_LINE INDENT y [ j ] [ i ] = y [ j + 1 ] [ i - 1 ] - y [ j ] [ i - 1 ] ; NEW_LINE DEDENT DEDENT
for i in range ( n ) : NEW_LINE INDENT for j in range ( n - i ) : NEW_LINE INDENT print ( y [ i ] [ j ] , " TABSYMBOL " , end = " ▁ " ) ; NEW_LINE DEDENT print ( " " ) ; NEW_LINE DEDENT
value = 27.4 ; NEW_LINE
sum = ( y [ 2 ] [ 0 ] + y [ 3 ] [ 0 ] ) / 2 ; NEW_LINE
k = 0 ; NEW_LINE
k = int ( n / 2 ) ; NEW_LINE else : NEW_LINE
u = ( value - x [ k ] ) / ( x [ 1 ] - x [ 0 ] ) ; NEW_LINE
for i in range ( 1 , n ) : NEW_LINE INDENT if ( i % 2 ) : NEW_LINE INDENT sum = sum + ( ( u - 0.5 ) * ucal ( u , i - 1 ) * y [ k ] [ i ] ) / fact ( i ) ; NEW_LINE DEDENT else : NEW_LINE INDENT sum = sum + ( ucal ( u , i ) * ( y [ k ] [ i ] + y [ k - 1 ] [ i ] ) / ( fact ( i ) * 2 ) ) ; NEW_LINE k -= 1 ; NEW_LINE DEDENT DEDENT print ( " Value ▁ at " , value , " is " , round ( sum , 5 ) ) ; NEW_LINE
def fibonacci ( n ) : NEW_LINE INDENT a = 0 NEW_LINE b = 1 NEW_LINE if ( n <= 1 ) : NEW_LINE INDENT return n NEW_LINE DEDENT for i in range ( 2 , n + 1 ) : NEW_LINE INDENT c = a + b NEW_LINE a = b NEW_LINE b = c NEW_LINE DEDENT return c NEW_LINE DEDENT
def isMultipleOf10 ( n ) : NEW_LINE INDENT f = fibonacci ( 30 ) NEW_LINE return ( f % 10 == 0 ) NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 30 NEW_LINE if ( isMultipleOf10 ( n ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT DEDENT
def powerof2 ( n ) : NEW_LINE
if n == 1 : NEW_LINE INDENT return True NEW_LINE DEDENT
elif n % 2 != 0 or n == 0 : NEW_LINE INDENT return False NEW_LINE DEDENT
return powerof2 ( n / 2 ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
INDENT print ( powerof2 ( 64 ) ) NEW_LINE DEDENT
INDENT print ( powerof2 ( 12 ) ) NEW_LINE DEDENT
def isPowerOfTwo ( x ) : NEW_LINE
return ( x and ( not ( x & ( x - 1 ) ) ) ) NEW_LINE
if ( isPowerOfTwo ( 31 ) ) : NEW_LINE INDENT print ( ' Yes ' ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( ' No ' ) NEW_LINE DEDENT if ( isPowerOfTwo ( 64 ) ) : NEW_LINE INDENT print ( ' Yes ' ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( ' No ' ) NEW_LINE DEDENT
def isPowerofTwo ( n ) : NEW_LINE INDENT if ( n == 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT if ( ( n & ( ~ ( n - 1 ) ) ) == n ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT return 0 NEW_LINE DEDENT
if ( isPowerofTwo ( 30 ) ) : NEW_LINE INDENT print ( ' Yes ' ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( ' No ' ) NEW_LINE DEDENT if ( isPowerofTwo ( 128 ) ) : NEW_LINE INDENT print ( ' Yes ' ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( ' No ' ) NEW_LINE DEDENT
def nextPowerOf2 ( n ) : NEW_LINE
p = 1 NEW_LINE
if ( n and not ( n & ( n - 1 ) ) ) : NEW_LINE INDENT return n NEW_LINE DEDENT
while ( p < n ) : NEW_LINE INDENT p <<= 1 NEW_LINE DEDENT return p NEW_LINE
def memoryUsed ( arr , n ) : NEW_LINE
sum = 0 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT sum += arr [ i ] NEW_LINE DEDENT
nearest = nextPowerOf2 ( sum ) NEW_LINE return nearest NEW_LINE
arr = [ 1 , 2 , 3 , 2 ] NEW_LINE n = len ( arr ) NEW_LINE print ( memoryUsed ( arr , n ) ) NEW_LINE
def toggleKthBit ( n , k ) : NEW_LINE INDENT return ( n ^ ( 1 << ( k - 1 ) ) ) NEW_LINE DEDENT
n = 5 NEW_LINE k = 1 NEW_LINE print ( toggleKthBit ( n , k ) ) NEW_LINE
def nextPowerOf2 ( n ) : NEW_LINE INDENT count = 0 NEW_LINE DEDENT
if ( n and not ( n & ( n - 1 ) ) ) : NEW_LINE INDENT return n NEW_LINE DEDENT while ( n != 0 ) : NEW_LINE INDENT n >>= 1 NEW_LINE count += 1 NEW_LINE DEDENT return 1 << count NEW_LINE
n = 0 NEW_LINE print ( nextPowerOf2 ( n ) ) NEW_LINE
def gcd ( A , B ) : NEW_LINE INDENT if ( B == 0 ) : NEW_LINE INDENT return A NEW_LINE DEDENT return gcd ( B , A % B ) NEW_LINE DEDENT
def lcm ( A , B ) : NEW_LINE INDENT return ( A * B ) // gcd ( A , B ) NEW_LINE DEDENT
def checkA ( A , B , C , K ) : NEW_LINE
start = 1 NEW_LINE end = K NEW_LINE
ans = - 1 NEW_LINE while ( start <= end ) : NEW_LINE INDENT mid = ( start + end ) // 2 NEW_LINE value = A * mid NEW_LINE divA = mid - 1 NEW_LINE divB = value // B - 1 if ( value % B == 0 ) else value // B NEW_LINE divC = value // C - 1 if ( value % C == 0 ) else value // C NEW_LINE divAB = value // lcm ( A , B ) - 1 if ( value % lcm ( A , B ) == 0 ) else value // lcm ( A , B ) NEW_LINE divBC = value // lcm ( C , B ) - 1 if ( value % lcm ( C , B ) == 0 ) else value // lcm ( C , B ) NEW_LINE divAC = value // lcm ( A , C ) - 1 if ( value % lcm ( A , C ) == 0 ) else value // lcm ( A , C ) NEW_LINE divABC = value // lcm ( A , lcm ( B , C ) ) - 1 if ( value % lcm ( A , lcm ( B , C ) ) == 0 ) else value // lcm ( A , lcm ( B , C ) ) NEW_LINE DEDENT
elem = divA + divB + divC - divAC - divBC - divAB + divABC NEW_LINE if ( elem == ( K - 1 ) ) : NEW_LINE INDENT ans = value NEW_LINE break NEW_LINE DEDENT
elif ( elem > ( K - 1 ) ) : NEW_LINE INDENT end = mid - 1 NEW_LINE DEDENT
else : NEW_LINE INDENT start = mid + 1 NEW_LINE DEDENT return ans NEW_LINE
def checkB ( A , B , C , K ) : NEW_LINE
start = 1 NEW_LINE end = K NEW_LINE
ans = - 1 NEW_LINE while ( start <= end ) : NEW_LINE INDENT mid = ( start + end ) // 2 NEW_LINE value = B * mid NEW_LINE divB = mid - 1 NEW_LINE if ( value % A == 0 ) : NEW_LINE INDENT divA = value // A - 1 NEW_LINE DEDENT else : value // A NEW_LINE if ( value % C == 0 ) : NEW_LINE INDENT divC = value // C - 1 NEW_LINE DEDENT else : value // C NEW_LINE if ( value % lcm ( A , B ) == 0 ) : NEW_LINE INDENT divAB = value // lcm ( A , B ) - 1 NEW_LINE DEDENT else : value // lcm ( A , B ) NEW_LINE if ( value % lcm ( C , B ) == 0 ) : NEW_LINE INDENT divBC = value // lcm ( C , B ) - 1 NEW_LINE DEDENT else : value // lcm ( C , B ) NEW_LINE if ( value % lcm ( A , C ) == 0 ) : NEW_LINE INDENT divAC = value // lcm ( A , C ) - 1 NEW_LINE DEDENT else : value // lcm ( A , C ) NEW_LINE if ( value % lcm ( A , lcm ( B , C ) ) == 0 ) : NEW_LINE INDENT divABC = value // lcm ( A , lcm ( B , C ) ) - 1 NEW_LINE DEDENT else : value // lcm ( A , lcm ( B , C ) ) NEW_LINE DEDENT
elem = divA + divB + divC - divAC - divBC - divAB + divABC NEW_LINE if ( elem == ( K - 1 ) ) : NEW_LINE INDENT ans = value NEW_LINE break NEW_LINE DEDENT
elif ( elem > ( K - 1 ) ) : NEW_LINE INDENT end = mid - 1 NEW_LINE DEDENT
else : NEW_LINE INDENT start = mid + 1 NEW_LINE DEDENT return ans NEW_LINE
def checkC ( A , B , C , K ) : NEW_LINE
start = 1 NEW_LINE end = K NEW_LINE
ans = - 1 NEW_LINE while ( start <= end ) : NEW_LINE INDENT mid = ( start + end ) // 2 NEW_LINE value = C * mid NEW_LINE divC = mid - 1 NEW_LINE if ( value % B == 0 ) : NEW_LINE INDENT divB = value // B - 1 NEW_LINE DEDENT else : value // B NEW_LINE if ( value % A == 0 ) : NEW_LINE INDENT divA = value // A - 1 NEW_LINE DEDENT else : value // A NEW_LINE if ( value % lcm ( A , B ) == 0 ) : NEW_LINE INDENT divAB = value // lcm ( A , B ) - 1 NEW_LINE DEDENT else : value // lcm ( A , B ) NEW_LINE if ( value % lcm ( C , B ) == 0 ) : NEW_LINE INDENT divBC = value // lcm ( C , B ) - 1 NEW_LINE DEDENT else : value // lcm ( C , B ) NEW_LINE if ( value % lcm ( A , C ) == 0 ) : NEW_LINE INDENT divAC = value // lcm ( A , C ) - 1 NEW_LINE DEDENT else : value // lcm ( A , C ) NEW_LINE if ( value % lcm ( A , lcm ( B , C ) ) == 0 ) : NEW_LINE INDENT divABC = value // lcm ( A , lcm ( B , C ) ) - 1 NEW_LINE DEDENT else : value // lcm ( A , lcm ( B , C ) ) NEW_LINE DEDENT
elem = divA + divB + divC - divAC - divBC - divAB + divABC NEW_LINE if ( elem == ( K - 1 ) ) : NEW_LINE INDENT ans = value NEW_LINE break NEW_LINE DEDENT
elif ( elem > ( K - 1 ) ) : NEW_LINE INDENT end = mid - 1 NEW_LINE DEDENT
else : NEW_LINE INDENT start = mid + 1 NEW_LINE DEDENT return ans NEW_LINE
def findKthMultiple ( A , B , C , K ) : NEW_LINE
res = checkA ( A , B , C , K ) NEW_LINE
if ( res == - 1 ) : NEW_LINE INDENT res = checkB ( A , B , C , K ) NEW_LINE DEDENT
if ( res == - 1 ) : NEW_LINE INDENT res = checkC ( A , B , C , K ) NEW_LINE DEDENT return res NEW_LINE
A = 2 NEW_LINE B = 4 NEW_LINE C = 5 NEW_LINE K = 5 NEW_LINE print ( findKthMultiple ( A , B , C , K ) ) NEW_LINE
def variationStalinsort ( arr ) : NEW_LINE INDENT j = 0 NEW_LINE while True : NEW_LINE INDENT moved = 0 NEW_LINE for i in range ( len ( arr ) - 1 - j ) : NEW_LINE INDENT if arr [ i ] > arr [ i + 1 ] : NEW_LINE DEDENT DEDENT DEDENT
arr . insert ( moved , arr . pop ( i + 1 ) ) NEW_LINE moved += 1 NEW_LINE j += 1 NEW_LINE if moved == 0 : NEW_LINE break NEW_LINE return arr NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 2 , 1 , 4 , 3 , 6 , 5 , 8 , 7 , 10 , 9 ] NEW_LINE DEDENT
print ( variationStalinsort ( arr ) ) NEW_LINE
def printArray ( arr , N ) : NEW_LINE
for i in range ( N ) : NEW_LINE INDENT print ( arr [ i ] , end = ' ▁ ' ) NEW_LINE DEDENT
def sortArray ( arr , N ) : NEW_LINE INDENT i = 0 NEW_LINE DEDENT
while i < N : NEW_LINE
if arr [ i ] == i + 1 : NEW_LINE INDENT i += 1 NEW_LINE DEDENT
else : NEW_LINE
temp1 = arr [ i ] NEW_LINE temp2 = arr [ arr [ i ] - 1 ] NEW_LINE arr [ i ] = temp2 NEW_LINE arr [ temp1 - 1 ] = temp1 NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 2 , 1 , 5 , 3 , 4 ] NEW_LINE N = len ( arr ) NEW_LINE DEDENT
sortArray ( arr , N ) NEW_LINE
printArray ( arr , N ) NEW_LINE
def maximum ( value , weight , weight1 , flag , K , index , val_len ) : NEW_LINE
if ( index >= val_len ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
if ( flag == K ) : NEW_LINE
skip = maximum ( value , weight , weight1 , flag , K , index + 1 , val_len ) NEW_LINE full = 0 NEW_LINE
if ( weight [ index ] <= weight1 ) : NEW_LINE INDENT full = value [ index ] + maximum ( value , weight , weight1 - weight [ index ] , flag , K , index + 1 , val_len ) NEW_LINE DEDENT
return max ( full , skip ) NEW_LINE
else : NEW_LINE
skip = maximum ( value , weight , weight1 , flag , K , index + 1 , val_len ) NEW_LINE full = 0 NEW_LINE half = 0 NEW_LINE
if ( weight [ index ] <= weight1 ) : NEW_LINE INDENT full = value [ index ] + maximum ( value , weight , weight1 - weight [ index ] , flag , K , index + 1 , val_len ) NEW_LINE DEDENT
if ( weight [ index ] / 2 <= weight1 ) : NEW_LINE INDENT half = value [ index ] + maximum ( value , weight , weight1 - weight [ index ] / 2 , flag , K , index + 1 , val_len ) NEW_LINE DEDENT
return max ( full , max ( skip , half ) ) NEW_LINE
value = [ 17 , 20 , 10 , 15 ] NEW_LINE weight = [ 4 , 2 , 7 , 5 ] NEW_LINE K = 1 NEW_LINE W = 4 NEW_LINE val_len = len ( value ) NEW_LINE print ( maximum ( value , weight , W , 0 , K , 0 , val_len ) ) NEW_LINE
N = 1005 NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def newNode ( data ) : NEW_LINE INDENT node = Node ( data ) NEW_LINE return node NEW_LINE DEDENT
dp = [ [ [ - 1 for i in range ( 5 ) ] for j in range ( 5 ) ] for k in range ( N ) ] ; NEW_LINE
def minDominatingSet ( root , covered , compulsory ) : NEW_LINE
if ( not root ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT
if ( not root . left and not root . right and not covered ) : NEW_LINE INDENT compulsory = True ; NEW_LINE DEDENT
if ( dp [ root . data ] [ covered ] [ compulsory ] != - 1 ) : NEW_LINE INDENT return dp [ root . data ] [ covered ] [ compulsory ] ; NEW_LINE DEDENT
if ( compulsory ) : NEW_LINE INDENT dp [ root . data ] [ covered ] [ compulsory ] = 1 + minDominatingSet ( root . left , 1 , 0 ) + minDominatingSet ( root . right , 1 , 0 ) ; NEW_LINE DEDENT
return dp [ root . data ] [ covered ] [ compulsory ] NEW_LINE
if ( covered ) : NEW_LINE INDENT dp [ root . data ] [ covered ] [ compulsory ] = min ( 1 + minDominatingSet ( root . left , 1 , 0 ) + minDominatingSet ( root . right , 1 , 0 ) , minDominatingSet ( root . left , 0 , 0 ) + minDominatingSet ( root . right , 0 , 0 ) ) ; NEW_LINE return dp [ root . data ] [ covered ] [ compulsory ] NEW_LINE DEDENT
ans = 1 + minDominatingSet ( root . left , 1 , 0 ) + minDominatingSet ( root . right , 1 , 0 ) ; NEW_LINE if ( root . left ) : NEW_LINE INDENT ans = min ( ans , minDominatingSet ( root . left , 0 , 1 ) + minDominatingSet ( root . right , 0 , 0 ) ) ; NEW_LINE DEDENT if ( root . right ) : NEW_LINE INDENT ans = min ( ans , minDominatingSet ( root . left , 0 , 0 ) + minDominatingSet ( root . right , 0 , 1 ) ) ; NEW_LINE DEDENT
dp [ root . data ] [ covered ] [ compulsory ] = ans ; NEW_LINE return ans NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
root = newNode ( 1 ) ; NEW_LINE root . left = newNode ( 2 ) ; NEW_LINE root . left . left = newNode ( 3 ) ; NEW_LINE root . left . right = newNode ( 4 ) ; NEW_LINE root . left . left . left = newNode ( 5 ) ; NEW_LINE root . left . left . left . left = newNode ( 6 ) ; NEW_LINE root . left . left . left . right = newNode ( 7 ) ; NEW_LINE root . left . left . left . right . right = newNode ( 10 ) ; NEW_LINE root . left . left . left . left . left = newNode ( 8 ) ; NEW_LINE root . left . left . left . left . right = newNode ( 9 ) ; NEW_LINE print ( minDominatingSet ( root , 0 , 0 ) ) NEW_LINE
import numpy as np NEW_LINE maxSum = 100 NEW_LINE arrSize = 51 NEW_LINE
dp = np . zeros ( ( arrSize , maxSum ) ) ; NEW_LINE visit = np . zeros ( ( arrSize , maxSum ) ) ; NEW_LINE
def SubsetCnt ( i , s , arr , n ) : NEW_LINE
if ( i == n ) : NEW_LINE INDENT if ( s == 0 ) : NEW_LINE INDENT return 1 ; NEW_LINE DEDENT else : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT DEDENT
if ( visit [ i ] [ s + arrSize ] ) : NEW_LINE INDENT return dp [ i ] [ s + arrSize ] ; NEW_LINE DEDENT
visit [ i ] [ s + arrSize ] = 1 ; NEW_LINE
dp [ i ] [ s + arrSize ] = ( SubsetCnt ( i + 1 , s + arr [ i ] , arr , n ) + SubsetCnt ( i + 1 , s , arr , n ) ) ; NEW_LINE
return dp [ i ] [ s + arrSize ] ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 2 , 2 , 2 , - 4 , - 4 ] ; NEW_LINE n = len ( arr ) ; NEW_LINE print ( SubsetCnt ( 0 , 0 , arr , n ) ) ; NEW_LINE DEDENT
def printTetra ( n ) : NEW_LINE INDENT if ( n < 0 ) : NEW_LINE INDENT return ; NEW_LINE DEDENT DEDENT
first = 0 ; NEW_LINE second = 1 ; NEW_LINE third = 1 ; NEW_LINE fourth = 2 ; NEW_LINE
curr = 0 ; NEW_LINE if ( n == 0 ) : NEW_LINE INDENT print ( first ) ; NEW_LINE DEDENT elif ( n == 1 or n == 2 ) : NEW_LINE INDENT print ( second ) ; NEW_LINE DEDENT elif ( n == 3 ) : NEW_LINE INDENT print ( fourth ) ; NEW_LINE DEDENT else : NEW_LINE
for i in range ( 4 , n + 1 ) : NEW_LINE INDENT curr = first + second + third + fourth ; NEW_LINE first = second ; NEW_LINE second = third ; NEW_LINE third = fourth ; NEW_LINE fourth = curr ; NEW_LINE DEDENT print ( curr ) ; NEW_LINE
n = 10 ; NEW_LINE printTetra ( n ) ; NEW_LINE
def countWays ( n ) : NEW_LINE INDENT res = [ 0 ] * ( n + 2 ) NEW_LINE res [ 0 ] = 1 NEW_LINE res [ 1 ] = 1 NEW_LINE res [ 2 ] = 2 NEW_LINE for i in range ( 3 , n + 1 ) : NEW_LINE INDENT res [ i ] = res [ i - 1 ] + res [ i - 2 ] + res [ i - 3 ] NEW_LINE DEDENT return res [ n ] NEW_LINE DEDENT
n = 4 NEW_LINE print ( countWays ( n ) ) NEW_LINE
def countWays ( n ) : NEW_LINE
a = 1 NEW_LINE b = 2 NEW_LINE c = 4 NEW_LINE
if ( n == 0 or n == 1 or n == 2 ) : NEW_LINE INDENT return n NEW_LINE DEDENT if ( n == 3 ) : NEW_LINE INDENT return c NEW_LINE DEDENT
for i in range ( 4 , n + 1 ) : NEW_LINE INDENT d = c + b + a NEW_LINE a = b NEW_LINE b = c NEW_LINE c = d NEW_LINE DEDENT return d NEW_LINE
n = 4 NEW_LINE print ( countWays ( n ) ) NEW_LINE
def isPossible ( elements , target ) : NEW_LINE INDENT dp = [ False ] * ( target + 1 ) NEW_LINE DEDENT
dp [ 0 ] = True NEW_LINE
for ele in elements : NEW_LINE
for j in range ( target , ele - 1 , - 1 ) : NEW_LINE INDENT if dp [ j - ele ] : NEW_LINE INDENT dp [ j ] = True NEW_LINE DEDENT DEDENT
return dp [ target ] NEW_LINE
arr = [ 6 , 2 , 5 ] NEW_LINE target = 7 NEW_LINE if isPossible ( arr , target ) : NEW_LINE INDENT print ( " YES " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NO " ) NEW_LINE DEDENT
def maxTasks ( high , low , n ) : NEW_LINE
if ( n <= 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
return max ( high [ n - 1 ] + maxTasks ( high , low , ( n - 2 ) ) , low [ n - 1 ] + maxTasks ( high , low , ( n - 1 ) ) ) ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 5 ; NEW_LINE high = [ 3 , 6 , 8 , 7 , 6 ] NEW_LINE low = [ 1 , 5 , 4 , 5 , 3 ] NEW_LINE print ( maxTasks ( high , low , n ) ) ; NEW_LINE DEDENT
import math NEW_LINE
def FindKthChar ( Str , K , X ) : NEW_LINE
ans = ' ▁ ' NEW_LINE Sum = 0 NEW_LINE
for i in range ( len ( Str ) ) : NEW_LINE
digit = ord ( Str [ i ] ) - 48 NEW_LINE
Range = int ( math . pow ( digit , X ) ) NEW_LINE Sum += Range NEW_LINE
if ( K <= Sum ) : NEW_LINE INDENT ans = Str [ i ] NEW_LINE break NEW_LINE DEDENT
return ans NEW_LINE
Str = "123" NEW_LINE K = 9 NEW_LINE X = 3 NEW_LINE
ans = FindKthChar ( Str , K , X ) NEW_LINE print ( ans ) NEW_LINE
def totalPairs ( s1 , s2 ) : NEW_LINE INDENT count = 0 ; NEW_LINE arr1 = [ 0 ] * 7 ; arr2 = [ 0 ] * 7 ; NEW_LINE DEDENT
for i in range ( len ( s1 ) ) : NEW_LINE INDENT set_bits = countSetBits ( ord ( s1 [ i ] ) ) NEW_LINE arr1 [ set_bits ] += 1 ; NEW_LINE DEDENT
for i in range ( len ( s2 ) ) : NEW_LINE INDENT set_bits = countSetBits ( ord ( s2 [ i ] ) ) ; NEW_LINE arr2 [ set_bits ] += 1 ; NEW_LINE DEDENT
for i in range ( 1 , 7 ) : NEW_LINE INDENT count += ( arr1 [ i ] * arr2 [ i ] ) ; NEW_LINE DEDENT
return count ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT s1 = " geeks " ; NEW_LINE s2 = " forgeeks " ; NEW_LINE print ( totalPairs ( s1 , s2 ) ) ; NEW_LINE DEDENT
def countSubstr ( str , n , x , y ) : NEW_LINE
tot_count = 0 NEW_LINE
count_x = 0 NEW_LINE
for i in range ( n ) : NEW_LINE
' NEW_LINE INDENT if str [ i ] == x : NEW_LINE INDENT count_x += 1 NEW_LINE DEDENT DEDENT
' NEW_LINE INDENT if str [ i ] == y : NEW_LINE INDENT tot_count += count_x NEW_LINE DEDENT DEDENT
return tot_count NEW_LINE
str = ' abbcaceghcak ' NEW_LINE n = len ( str ) NEW_LINE x , y = ' a ' , ' c ' NEW_LINE print ( ' Count ▁ = ' , countSubstr ( str , n , x , y ) ) NEW_LINE
OUT = 0 NEW_LINE IN = 1 NEW_LINE
def countWords ( string ) : NEW_LINE INDENT state = OUT NEW_LINE DEDENT
wc = 0 NEW_LINE
for i in range ( len ( string ) ) : NEW_LINE
if ( string [ i ] == ' ▁ ' or string [ i ] ==   ' ' ▁ or ▁ string [ i ] ▁ = = ▁ ' 	 ' ) : NEW_LINE INDENT state = OUT NEW_LINE DEDENT
elif state == OUT : NEW_LINE INDENT state = IN NEW_LINE wc += 1 NEW_LINE DEDENT
return wc NEW_LINE
string =   " One two three NEW_LINE INDENT four five   " NEW_LINE DEDENT print ( " No . ▁ of ▁ words ▁ : ▁ " + str ( countWords ( string ) ) ) NEW_LINE
def nthEnneadecagonal ( n ) : NEW_LINE
return ( 17 * n * n - 15 * n ) // 2 NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 6 NEW_LINE print ( n , " th ▁ Enneadecagonal ▁ number ▁ : " , nthEnneadecagonal ( n ) ) NEW_LINE DEDENT
PI = 3.14159265 NEW_LINE
def areacircumscribed ( a ) : NEW_LINE INDENT return ( a * a * ( PI / 2 ) ) NEW_LINE DEDENT
a = 6 NEW_LINE print ( " ▁ Area ▁ of ▁ an ▁ circumscribed ▁ circle ▁ is ▁ : " , round ( areacircumscribed ( a ) , 2 ) ) NEW_LINE
def itemType ( n ) : NEW_LINE
count = 0 NEW_LINE day = 1 NEW_LINE
while ( count + day * ( day + 1 ) // 2 < n ) : NEW_LINE
count += day * ( day + 1 ) // 2 ; NEW_LINE day += 1 NEW_LINE type = day NEW_LINE while ( type > 0 ) : NEW_LINE
count += type NEW_LINE
if ( count >= n ) : NEW_LINE INDENT return type NEW_LINE DEDENT type -= 1 NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 10 NEW_LINE print ( itemType ( N ) ) NEW_LINE DEDENT
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data ; NEW_LINE self . next = next ; NEW_LINE DEDENT DEDENT
def isSortedDesc ( head ) : NEW_LINE INDENT if ( head == None ) : NEW_LINE INDENT return True ; NEW_LINE DEDENT DEDENT
while ( head . next != None ) : NEW_LINE INDENT t = head ; NEW_LINE if ( t . data <= t . next . data ) : NEW_LINE INDENT return False ; NEW_LINE DEDENT head = head . next NEW_LINE DEDENT return True ; NEW_LINE def newNode ( data ) : NEW_LINE temp = Node ( 0 ) ; NEW_LINE temp . next = None ; NEW_LINE temp . data = data ; NEW_LINE return temp ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT head = newNode ( 7 ) ; NEW_LINE head . next = newNode ( 5 ) ; NEW_LINE head . next . next = newNode ( 4 ) ; NEW_LINE head . next . next . next = newNode ( 3 ) ; NEW_LINE if ( isSortedDesc ( head ) ) : NEW_LINE INDENT print ( " Yes " ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) ; NEW_LINE DEDENT DEDENT
def maxLength ( str , n , c , k ) : NEW_LINE
ans = - 1 NEW_LINE
' NEW_LINE INDENT cnt = 0 NEW_LINE DEDENT
left = 0 NEW_LINE for right in range ( 0 , n ) : NEW_LINE INDENT if ( str [ right ] == c ) : NEW_LINE INDENT cnt += 1 NEW_LINE DEDENT DEDENT
while ( cnt > k ) : NEW_LINE INDENT if ( str [ left ] == c ) : NEW_LINE INDENT cnt -= 1 NEW_LINE DEDENT DEDENT
left += 1 NEW_LINE
ans = max ( ans , right - left + 1 ) NEW_LINE return ans NEW_LINE
def maxConsecutiveSegment ( S , K ) : NEW_LINE INDENT N = len ( S ) NEW_LINE DEDENT
return max ( maxLength ( S , N , '0' , K ) , maxLength ( S , N , '1' , K ) ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT S = "1001" NEW_LINE K = 1 NEW_LINE print ( maxConsecutiveSegment ( S , K ) ) NEW_LINE DEDENT
def find ( N ) : NEW_LINE
F = int ( ( N - 4 ) / 5 ) NEW_LINE
if ( ( N - 5 * F ) % 2 ) == 0 : NEW_LINE INDENT O = 2 NEW_LINE DEDENT else : NEW_LINE INDENT O = 1 NEW_LINE DEDENT
T = ( N - 5 * F - O ) // 2 NEW_LINE print ( " Count ▁ of ▁ 5 ▁ valueds ▁ coins : ▁ " , F ) NEW_LINE print ( " Count ▁ of ▁ 2 ▁ valueds ▁ coins : ▁ " , T ) NEW_LINE print ( " Count ▁ of ▁ 1 ▁ valueds ▁ coins : ▁ " , O ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 8 NEW_LINE find ( N ) NEW_LINE DEDENT
' NEW_LINE def findMaxOccurence ( str , N ) : NEW_LINE
for i in range ( N ) : NEW_LINE
' NEW_LINE INDENT if ( str [ i ] == ' ? ' ) : NEW_LINE DEDENT
str = list ( "10?0?11" ) NEW_LINE N = len ( str ) NEW_LINE findMaxOccurence ( str , N ) NEW_LINE
def checkInfinite ( s ) : NEW_LINE
flag = 1 NEW_LINE N = len ( s ) NEW_LINE
for i in range ( N - 1 ) : NEW_LINE
if ( s [ i ] == chr ( ord ( s [ i + 1 ] ) + 1 ) ) : NEW_LINE INDENT continue NEW_LINE DEDENT
elif ( s [ i ] == ' a ' and s [ i + 1 ] == ' z ' ) : NEW_LINE INDENT continue NEW_LINE DEDENT
else : NEW_LINE INDENT flag = 0 NEW_LINE break NEW_LINE DEDENT
if ( flag == 0 ) : NEW_LINE INDENT print ( " NO " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " YES " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE
s = " ecbaz " NEW_LINE
checkInfinite ( s ) NEW_LINE
def minChangeInLane ( barrier , n ) : NEW_LINE INDENT dp = [ 1 , 0 , 1 ] NEW_LINE for j in range ( n ) : NEW_LINE DEDENT
val = barrier [ j ] NEW_LINE if ( val > 0 ) : NEW_LINE INDENT dp [ val - 1 ] = 1000000 NEW_LINE DEDENT for i in range ( 3 ) : NEW_LINE
if ( val != i + 1 ) : NEW_LINE INDENT dp [ i ] = min ( dp [ i ] , min ( dp [ ( i + 1 ) % 3 ] , dp [ ( i + 2 ) % 3 ] ) + 1 ) NEW_LINE DEDENT
return min ( dp [ 0 ] , min ( dp [ 1 ] , dp [ 2 ] ) ) NEW_LINE
barrier = [ 0 , 1 , 2 , 3 , 0 ] NEW_LINE N = len ( barrier ) NEW_LINE print ( minChangeInLane ( barrier , N ) ) NEW_LINE
def numWays ( ratings , queries , n , k ) : NEW_LINE
dp = [ [ 0 for i in range ( 10002 ) ] for j in range ( n ) ] ; NEW_LINE
for i in range ( k ) : NEW_LINE INDENT dp [ 0 ] [ ratings [ 0 ] [ i ] ] += 1 ; NEW_LINE DEDENT
for i in range ( 1 , n ) : NEW_LINE
for sum in range ( 10001 ) : NEW_LINE
for j in range ( k ) : NEW_LINE
if ( sum >= ratings [ i ] [ j ] ) : NEW_LINE INDENT dp [ i ] [ sum ] += dp [ i - 1 ] [ sum - ratings [ i ] [ j ] ] ; NEW_LINE DEDENT
for sum in range ( 1 , 10001 ) : NEW_LINE INDENT dp [ n - 1 ] [ sum ] += dp [ n - 1 ] [ sum - 1 ] ; NEW_LINE DEDENT
for q in range ( len ( queries ) ) : NEW_LINE INDENT a = queries [ q ] [ 0 ] ; NEW_LINE b = queries [ q ] [ 1 ] ; NEW_LINE DEDENT
print ( dp [ n - 1 ] [ b ] - dp [ n - 1 ] [ a - 1 ] , end = " ▁ " ) ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
N = 2 ; NEW_LINE K = 3 ; NEW_LINE
ratings = [ [ 1 , 2 , 3 ] , [ 4 , 5 , 6 ] ] ; NEW_LINE queries = [ [ 6 , 6 ] , [ 1 , 6 ] ] ; NEW_LINE
numWays ( ratings , queries , N , K ) ; NEW_LINE
def numberOfPermWithKInversion ( N , K ) : NEW_LINE
dp = [ [ 0 ] * ( K + 1 ) ] * 2 NEW_LINE mod = 1000000007 NEW_LINE for i in range ( 1 , N + 1 ) : NEW_LINE INDENT for j in range ( 0 , K + 1 ) : NEW_LINE DEDENT
if ( i == 1 ) : NEW_LINE INDENT dp [ i % 2 ] [ j ] = 1 if ( j == 0 ) else 0 NEW_LINE DEDENT
elif ( j == 0 ) : NEW_LINE INDENT dp [ i % 2 ] [ j ] = 1 NEW_LINE DEDENT
else : NEW_LINE INDENT var = ( 0 if ( max ( j - ( i - 1 ) , 0 ) == 0 ) else dp [ 1 - i % 2 ] [ max ( j - ( i - 1 ) , 0 ) - 1 ] ) NEW_LINE dp [ i % 2 ] [ j ] = ( ( dp [ i % 2 ] [ j - 1 ] % mod + ( dp [ 1 - i % 2 ] [ j ] - ( var ) + mod ) % mod ) % mod ) NEW_LINE DEDENT
print ( dp [ N % 2 ] [ K ] ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
N = 3 NEW_LINE K = 2 NEW_LINE
numberOfPermWithKInversion ( N , K ) NEW_LINE
def MaxProfit ( treasure , color , n , k , col , A , B ) : NEW_LINE INDENT sum = 0 NEW_LINE DEDENT
if k == n : NEW_LINE INDENT return 0 NEW_LINE DEDENT
if col == color [ k ] : NEW_LINE INDENT sum += max ( A * treasure [ k ] + MaxProfit ( treasure , color , n , k + 1 , color [ k ] , A , B ) , MaxProfit ( treasure , color , n , k + 1 , col , A , B ) ) NEW_LINE DEDENT else : NEW_LINE INDENT sum += max ( B * treasure [ k ] + MaxProfit ( treasure , color , n , k + 1 , color [ k ] , A , B ) , MaxProfit ( treasure , color , n , k + 1 , col , A , B ) ) NEW_LINE DEDENT
return sum NEW_LINE
A = - 5 NEW_LINE B = 7 NEW_LINE treasure = [ 4 , 8 , 2 , 9 ] NEW_LINE color = [ 2 , 2 , 6 , 2 ] NEW_LINE n = len ( color ) NEW_LINE
print ( MaxProfit ( treasure , color , n , 0 , 0 , A , B ) ) NEW_LINE
def printTetraRec ( n ) : NEW_LINE
if ( n == 0 ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT
if ( n == 1 or n == 2 ) : NEW_LINE INDENT return 1 ; NEW_LINE DEDENT
if ( n == 3 ) : NEW_LINE INDENT return 2 ; NEW_LINE DEDENT else : NEW_LINE INDENT return ( printTetraRec ( n - 1 ) + printTetraRec ( n - 2 ) + printTetraRec ( n - 3 ) + printTetraRec ( n - 4 ) ) ; NEW_LINE DEDENT
def printTetra ( n ) : NEW_LINE INDENT print ( printTetraRec ( n ) , end = " ▁ " ) ; NEW_LINE DEDENT
n = 10 ; NEW_LINE printTetra ( n ) ; NEW_LINE
def Combination ( a , combi , n , r , depth , index ) : NEW_LINE INDENT global Sum NEW_LINE DEDENT
if index == r : NEW_LINE
product = 1 NEW_LINE for i in range ( r ) : NEW_LINE INDENT product = product * combi [ i ] NEW_LINE DEDENT
Sum += product NEW_LINE return NEW_LINE
for i in range ( depth , n ) : NEW_LINE INDENT combi [ index ] = a [ i ] NEW_LINE Combination ( a , combi , n , r , i + 1 , index + 1 ) NEW_LINE DEDENT
def allCombination ( a , n ) : NEW_LINE INDENT global Sum NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE DEDENT
combi = [ 0 ] * i NEW_LINE
Combination ( a , combi , n , i , 0 , 0 ) NEW_LINE
print ( " f ( " , i , " ) ▁ - - > ▁ " , Sum ) NEW_LINE Sum = 0 NEW_LINE
Sum = 0 NEW_LINE n = 5 NEW_LINE a = [ 0 ] * n NEW_LINE
for i in range ( n ) : NEW_LINE INDENT a [ i ] = i + 1 NEW_LINE DEDENT
allCombination ( a , n ) NEW_LINE
/ * Returns the maximum among the 2 numbers * / NEW_LINE def max1 ( x , y ) : NEW_LINE INDENT return x if ( x > y ) else y ; NEW_LINE DEDENT
def maxTasks ( high , low , n ) : NEW_LINE
task_dp = [ 0 ] * ( n + 1 ) ; NEW_LINE
task_dp [ 0 ] = 0 ; NEW_LINE
task_dp [ 1 ] = high [ 0 ] ; NEW_LINE
for i in range ( 2 , n + 1 ) : NEW_LINE INDENT task_dp [ i ] = max ( high [ i - 1 ] + task_dp [ i - 2 ] , low [ i - 1 ] + task_dp [ i - 1 ] ) ; NEW_LINE DEDENT return task_dp [ n ] ; NEW_LINE
n = 5 ; NEW_LINE high = [ 3 , 6 , 8 , 7 , 6 ] ; NEW_LINE low = [ 1 , 5 , 4 , 5 , 3 ] ; NEW_LINE print ( maxTasks ( high , low , n ) ) ; NEW_LINE
n = 10 NEW_LINE k = 2 NEW_LINE print ( " Value ▁ of ▁ P ( " , n , " , ▁ " , k , " ) ▁ is ▁ " , permutationCoeff ( n , k ) ) NEW_LINE
def findPartition ( arr , n ) : NEW_LINE INDENT sum = 0 NEW_LINE i , j = 0 , 0 NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT sum += arr [ i ] NEW_LINE DEDENT if sum % 2 != 0 : NEW_LINE INDENT return false NEW_LINE DEDENT part = [ [ True for i in range ( n + 1 ) ] for j in range ( sum // 2 + 1 ) ] NEW_LINE
for i in range ( 0 , n + 1 ) : NEW_LINE INDENT part [ 0 ] [ i ] = True NEW_LINE DEDENT
for i in range ( 1 , sum // 2 + 1 ) : NEW_LINE INDENT part [ i ] [ 0 ] = False NEW_LINE DEDENT
for i in range ( 1 , sum // 2 + 1 ) : NEW_LINE INDENT for j in range ( 1 , n + 1 ) : NEW_LINE INDENT part [ i ] [ j ] = part [ i ] [ j - 1 ] NEW_LINE if i >= arr [ j - 1 ] : NEW_LINE INDENT part [ i ] [ j ] = ( part [ i ] [ j ] or part [ i - arr [ j - 1 ] ] [ j - 1 ] ) NEW_LINE DEDENT DEDENT DEDENT
return part [ sum // 2 ] [ n ] NEW_LINE
arr = [ 3 , 1 , 1 , 2 , 2 , 1 ] NEW_LINE n = len ( arr ) NEW_LINE
if findPartition ( arr , n ) == True : NEW_LINE INDENT print ( " Can ▁ be ▁ divided ▁ into ▁ two " , " subsets ▁ of ▁ equal ▁ sum " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Can ▁ not ▁ be ▁ divided ▁ into ▁ " , " two ▁ subsets ▁ of ▁ equal ▁ sum " ) NEW_LINE DEDENT
def minimumOperations ( orig_str , m , n ) : NEW_LINE
orig = orig_str NEW_LINE
turn = 1 NEW_LINE j = 1 NEW_LINE
for i in orig_str : NEW_LINE
m_cut = orig_str [ - m : ] NEW_LINE orig_str = orig_str . replace ( ' ▁ ' , ' ' ) [ : - m ] NEW_LINE
orig_str = m_cut + orig_str NEW_LINE
j = j + 1 NEW_LINE
if orig != orig_str : NEW_LINE INDENT turn = turn + 1 NEW_LINE DEDENT
n_cut = orig_str [ - n : ] NEW_LINE orig_str = orig_str . replace ( ' ▁ ' , ' ' ) [ : - n ] NEW_LINE
orig_str = n_cut + orig_str NEW_LINE
j = j + 1 NEW_LINE
if orig == orig_str : NEW_LINE INDENT break NEW_LINE DEDENT
turn = turn + 1 NEW_LINE print ( turn ) NEW_LINE
S = " GeeksforGeeks " NEW_LINE X = 5 NEW_LINE Y = 3 NEW_LINE
minimumOperations ( S , X , Y ) NEW_LINE
def KMPSearch ( pat , txt ) : NEW_LINE INDENT M = len ( pat ) NEW_LINE N = len ( txt ) NEW_LINE DEDENT
lps = [ 0 ] * M NEW_LINE
computeLPSArray ( pat , M , lps ) NEW_LINE
i = 0 NEW_LINE j = 0 NEW_LINE while i < N : NEW_LINE INDENT if pat [ j ] == txt [ i ] : NEW_LINE INDENT j += 1 NEW_LINE i += 1 NEW_LINE DEDENT if j == M : NEW_LINE INDENT return i - j NEW_LINE j = lps [ j - 1 ] NEW_LINE DEDENT DEDENT
elif i < N and pat [ j ] != txt [ i ] : NEW_LINE
if j != 0 : NEW_LINE INDENT j = lps [ j - 1 ] NEW_LINE DEDENT else : NEW_LINE INDENT i = i + 1 NEW_LINE DEDENT
def computeLPSArray ( pat , M , lps ) : NEW_LINE
_len = 0 NEW_LINE
lps [ 0 ] = 0 NEW_LINE
i = 1 NEW_LINE while i < M : NEW_LINE INDENT if pat [ i ] == pat [ _len ] : NEW_LINE INDENT _len += 1 NEW_LINE lps [ i ] = _len NEW_LINE i += 1 NEW_LINE DEDENT DEDENT
else : NEW_LINE
if _len != 0 : NEW_LINE INDENT _len = lps [ _len - 1 ] NEW_LINE DEDENT else : NEW_LINE INDENT lps [ i ] = 0 NEW_LINE i += 1 NEW_LINE DEDENT
def countRotations ( s ) : NEW_LINE
s1 = s [ 1 : len ( s ) ] + s NEW_LINE
pat = s [ : ] NEW_LINE text = s1 [ : ] NEW_LINE
return 1 + KMPSearch ( pat , text ) NEW_LINE
s1 = " geeks " NEW_LINE print ( countRotations ( s1 ) ) NEW_LINE
def start ( c ) : NEW_LINE
if ( c == ' t ' or c == ' T ' ) : NEW_LINE INDENT dfa = 1 NEW_LINE DEDENT
def state1 ( c ) : NEW_LINE
if ( c == ' t ' or c == ' T ' ) : NEW_LINE INDENT dfa = 1 NEW_LINE DEDENT
elif ( c == ' h ' or c == ' H ' ) : NEW_LINE INDENT dfa = 2 NEW_LINE DEDENT
else : NEW_LINE INDENT dfa = 0 NEW_LINE DEDENT
def state2 ( c ) : NEW_LINE
if ( c == ' e ' or c == ' E ' ) : NEW_LINE INDENT dfa = 3 NEW_LINE DEDENT else : NEW_LINE INDENT dfa = 0 NEW_LINE DEDENT
def state3 ( c ) : NEW_LINE
if ( c == ' t ' or c == ' T ' ) : NEW_LINE INDENT dfa = 1 NEW_LINE DEDENT else : NEW_LINE INDENT dfa = 0 NEW_LINE DEDENT def isAccepted ( string ) : NEW_LINE
length = len ( string ) NEW_LINE for i in range ( length ) : NEW_LINE INDENT if ( dfa == 0 ) : NEW_LINE INDENT start ( string [ i ] ) NEW_LINE DEDENT elif ( dfa == 1 ) : NEW_LINE INDENT state1 ( string [ i ] ) NEW_LINE DEDENT elif ( dfa == 2 ) : NEW_LINE INDENT state2 ( string [ i ] ) NEW_LINE DEDENT else : NEW_LINE INDENT state3 ( string [ i ] ) NEW_LINE DEDENT DEDENT return ( dfa != 3 ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT string = " forTHEgeeks " NEW_LINE DEDENT dfa = 0 NEW_LINE INDENT if isAccepted ( string ) : NEW_LINE INDENT print ( " ACCEPTED " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NOT ▁ ACCEPTED " ) NEW_LINE DEDENT DEDENT
parent = [ 0 ] * 256 NEW_LINE
def find ( x ) : NEW_LINE INDENT if ( x != parent [ x ] ) : NEW_LINE INDENT parent [ x ] = find ( parent [ x ] ) NEW_LINE return parent [ x ] NEW_LINE DEDENT return x NEW_LINE DEDENT
def join ( x , y ) : NEW_LINE INDENT px = find ( x ) NEW_LINE pz = find ( y ) NEW_LINE if ( px != pz ) : NEW_LINE INDENT parent [ pz ] = px NEW_LINE DEDENT DEDENT
def convertible ( s1 , s2 ) : NEW_LINE
mp = dict ( ) NEW_LINE for i in range ( len ( s1 ) ) : NEW_LINE INDENT if ( s1 [ i ] in mp ) : NEW_LINE INDENT mp [ s1 [ i ] ] = s2 [ i ] NEW_LINE DEDENT else : NEW_LINE INDENT if s1 [ i ] in mp and mp [ s1 [ i ] ] != s2 [ i ] : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT DEDENT
for it in mp : NEW_LINE INDENT if ( it == mp [ it ] ) : NEW_LINE INDENT continue NEW_LINE DEDENT else : NEW_LINE INDENT if ( find ( ord ( it ) ) == find ( ord ( it ) ) ) : NEW_LINE INDENT return False NEW_LINE DEDENT else : NEW_LINE INDENT join ( ord ( it ) , ord ( it ) ) NEW_LINE DEDENT DEDENT DEDENT return True NEW_LINE
def initialize ( ) : NEW_LINE INDENT for i in range ( 256 ) : NEW_LINE INDENT parent [ i ] = i NEW_LINE DEDENT DEDENT
s1 = " abbcaa " NEW_LINE s2 = " bccdbb " NEW_LINE initialize ( ) NEW_LINE if ( convertible ( s1 , s2 ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
SIZE = 26 NEW_LINE from math import sqrt NEW_LINE
def SieveOfEratosthenes ( prime , p_size ) : NEW_LINE
prime [ 0 ] = False NEW_LINE prime [ 1 ] = False NEW_LINE for p in range ( 2 , int ( sqrt ( p_size ) ) , 1 ) : NEW_LINE
if ( prime [ p ] ) : NEW_LINE
for i in range ( p * 2 , p_size , p ) : NEW_LINE INDENT prime [ i ] = False NEW_LINE DEDENT
def printChar ( str , n ) : NEW_LINE INDENT prime = [ True for i in range ( n + 1 ) ] NEW_LINE DEDENT
SieveOfEratosthenes ( prime , len ( str ) + 1 ) NEW_LINE
freq = [ 0 for i in range ( SIZE ) ] NEW_LINE
for i in range ( n ) : NEW_LINE INDENT freq [ ord ( str [ i ] ) - ord ( ' a ' ) ] += 1 NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE
if ( prime [ freq [ ord ( str [ i ] ) - ord ( ' a ' ) ] ] ) : NEW_LINE INDENT print ( str [ i ] , end = " " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT str = " geeksforgeeks " NEW_LINE n = len ( str ) NEW_LINE printChar ( str , n ) NEW_LINE DEDENT
from collections import Counter NEW_LINE import math NEW_LINE
def prime ( n ) : NEW_LINE INDENT if n <= 1 : NEW_LINE INDENT return False NEW_LINE DEDENT max_div = math . floor ( math . sqrt ( n ) ) NEW_LINE for i in range ( 2 , 1 + max_div ) : NEW_LINE INDENT if n % i == 0 : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT return True NEW_LINE DEDENT def checkString ( s ) : NEW_LINE
freq = Counter ( s ) NEW_LINE
for i in range ( len ( s ) ) : NEW_LINE INDENT if prime ( freq [ s [ i ] ] ) : NEW_LINE INDENT print ( s [ i ] , end = " " ) NEW_LINE DEDENT DEDENT
s = " geeksforgeeks " NEW_LINE
checkString ( s ) NEW_LINE
SIZE = 26 NEW_LINE
def printChar ( string , n ) : NEW_LINE
freq = [ 0 ] * SIZE NEW_LINE
for i in range ( 0 , n ) : NEW_LINE INDENT freq [ ord ( string [ i ] ) - ord ( ' a ' ) ] += 1 NEW_LINE DEDENT
for i in range ( 0 , n ) : NEW_LINE
if ( freq [ ord ( string [ i ] ) - ord ( ' a ' ) ] % 2 == 0 ) : NEW_LINE INDENT print ( string [ i ] , end = " " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT string = " geeksforgeeks " NEW_LINE n = len ( string ) NEW_LINE printChar ( string , n ) NEW_LINE DEDENT
def CompareAlphanumeric ( str1 , str2 ) : NEW_LINE
i = 0 NEW_LINE j = 0 NEW_LINE
len1 = len ( str1 ) NEW_LINE
len2 = len ( str2 ) NEW_LINE
while ( i <= len1 and j <= len2 ) : NEW_LINE
while ( i < len1 and ( ( ( str1 [ i ] >= ' a ' and str1 [ i ] <= ' z ' ) or ( str1 [ i ] >= ' A ' and str1 [ i ] <= ' Z ' ) or ( str1 [ i ] >= '0' and str1 [ i ] <= '9' ) ) == False ) ) : NEW_LINE INDENT i += 1 NEW_LINE DEDENT
while ( j < len2 and ( ( ( str2 [ j ] >= ' a ' and str2 [ j ] <= ' z ' ) or ( str2 [ j ] >= ' A ' and str2 [ j ] <= ' Z ' ) or ( str2 [ j ] >= '0' and str2 [ j ] <= '9' ) ) == False ) ) : NEW_LINE INDENT j += 1 NEW_LINE DEDENT
if ( i == len1 and j == len2 ) : NEW_LINE INDENT return True NEW_LINE DEDENT
elif ( str1 [ i ] != str2 [ j ] ) : NEW_LINE INDENT return False NEW_LINE DEDENT
else : NEW_LINE INDENT i += 1 NEW_LINE j += 1 NEW_LINE DEDENT
return False NEW_LINE
def CompareAlphanumericUtil ( str1 , str2 ) : NEW_LINE
res = CompareAlphanumeric ( str1 , str2 ) NEW_LINE
if ( res == True ) : NEW_LINE INDENT print ( " Equal " ) NEW_LINE DEDENT
else : NEW_LINE INDENT print ( " Unequal " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT str1 = " Ram , ▁ Shyam " NEW_LINE str2 = " ▁ Ram ▁ - ▁ Shyam . " NEW_LINE CompareAlphanumericUtil ( str1 , str2 ) NEW_LINE str1 = " abc123" NEW_LINE str2 = "123abc " NEW_LINE CompareAlphanumericUtil ( str1 , str2 ) NEW_LINE DEDENT
def solveQueries ( Str , query ) : NEW_LINE
ll = len ( Str ) NEW_LINE
Q = len ( query ) NEW_LINE
pre = [ [ 0 for i in range ( 256 ) ] for i in range ( ll ) ] NEW_LINE
for i in range ( ll ) : NEW_LINE
pre [ i ] [ ord ( Str [ i ] ) ] += 1 NEW_LINE
if ( i ) : NEW_LINE
for j in range ( 256 ) : NEW_LINE INDENT pre [ i ] [ j ] += pre [ i - 1 ] [ j ] NEW_LINE DEDENT
for i in range ( Q ) : NEW_LINE
l = query [ i ] [ 0 ] NEW_LINE r = query [ i ] [ 1 ] NEW_LINE maxi = 0 NEW_LINE c = ' a ' NEW_LINE
for j in range ( 256 ) : NEW_LINE
times = pre [ r ] [ j ] NEW_LINE
if ( l ) : NEW_LINE INDENT times -= pre [ l - 1 ] [ j ] NEW_LINE DEDENT
if ( times > maxi ) : NEW_LINE INDENT maxi = times NEW_LINE c = chr ( j ) NEW_LINE DEDENT
print ( " Query ▁ " , i + 1 , " : ▁ " , c ) NEW_LINE
Str = " striver " NEW_LINE query = [ [ 0 , 1 ] , [ 1 , 6 ] , [ 5 , 6 ] ] NEW_LINE solveQueries ( Str , query ) NEW_LINE
def startsWith ( str , pre ) : NEW_LINE INDENT strLen = len ( str ) NEW_LINE preLen = len ( pre ) NEW_LINE i = 0 NEW_LINE j = 0 NEW_LINE DEDENT
while ( i < strLen and j < preLen ) : NEW_LINE
if ( str [ i ] != pre [ j ] ) : NEW_LINE INDENT return False NEW_LINE DEDENT i += 1 NEW_LINE j += 1 NEW_LINE
return True NEW_LINE
def endsWith ( str , suff ) : NEW_LINE INDENT i = len ( str ) - 1 NEW_LINE j = len ( suff ) - 1 NEW_LINE DEDENT
while ( i >= 0 and j >= 0 ) : NEW_LINE
if ( str [ i ] != suff [ j ] ) : NEW_LINE INDENT return False NEW_LINE DEDENT i -= 1 NEW_LINE j -= 1 NEW_LINE
return True NEW_LINE
def checkString ( str , a , b ) : NEW_LINE
if ( len ( str ) != len ( a ) + len ( b ) ) : NEW_LINE INDENT return False NEW_LINE DEDENT
if ( startsWith ( str , a ) ) : NEW_LINE
if ( endsWith ( str , b ) ) : NEW_LINE INDENT return True NEW_LINE DEDENT
if ( startsWith ( str , b ) ) : NEW_LINE
if ( endsWith ( str , a ) ) : NEW_LINE INDENT return True NEW_LINE DEDENT return False NEW_LINE
str = " GeeksforGeeks " NEW_LINE a = " Geeksfo " NEW_LINE b = " rGeeks " NEW_LINE if ( checkString ( str , a , b ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
import sys NEW_LINE import math NEW_LINE
def printChar ( str_ , n ) : NEW_LINE
freq = [ 0 ] * 26 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT freq [ ord ( str_ [ i ] ) - ord ( ' a ' ) ] += 1 NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE
if ( freq [ ord ( str_ [ i ] ) - ord ( ' a ' ) ] ) % 2 == 1 : NEW_LINE INDENT print ( " { } " . format ( str_ [ i ] ) , end = " " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT str_ = " geeksforgeeks " NEW_LINE n = len ( str_ ) NEW_LINE printChar ( str_ , n ) NEW_LINE DEDENT
def minOperations ( str , n ) : NEW_LINE
lastUpper = - 1 NEW_LINE firstLower = - 1 NEW_LINE
for i in range ( n - 1 , - 1 , - 1 ) : NEW_LINE INDENT if ( str [ i ] . isupper ( ) ) : NEW_LINE INDENT lastUpper = i NEW_LINE break NEW_LINE DEDENT DEDENT
for i in range ( n ) : NEW_LINE INDENT if ( str [ i ] . islower ( ) ) : NEW_LINE INDENT firstLower = i NEW_LINE break NEW_LINE DEDENT DEDENT
if ( lastUpper == - 1 or firstLower == - 1 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
countUpper = 0 NEW_LINE for i in range ( firstLower , n ) : NEW_LINE INDENT if ( str [ i ] . isupper ( ) ) : NEW_LINE INDENT countUpper += 1 NEW_LINE DEDENT DEDENT
countLower = 0 NEW_LINE for i in range ( lastUpper ) : NEW_LINE INDENT if ( str [ i ] . islower ( ) ) : NEW_LINE INDENT countLower += 1 NEW_LINE DEDENT DEDENT
return min ( countLower , countUpper ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT str = " geEksFOrGEekS " NEW_LINE n = len ( str ) NEW_LINE print ( minOperations ( str , n ) ) NEW_LINE DEDENT
import math NEW_LINE
def Betrothed_Sum ( n ) : NEW_LINE
Set = [ ] NEW_LINE for number_1 in range ( 1 , n ) : NEW_LINE
sum_divisor_1 = 1 NEW_LINE
i = 2 NEW_LINE while i * i <= number_1 : NEW_LINE INDENT if ( number_1 % i == 0 ) : NEW_LINE INDENT sum_divisor_1 = sum_divisor_1 + i NEW_LINE if ( i * i != number_1 ) : NEW_LINE INDENT sum_divisor_1 += number_1 // i NEW_LINE DEDENT DEDENT i = i + 1 NEW_LINE DEDENT if ( sum_divisor_1 > number_1 ) : NEW_LINE INDENT number_2 = sum_divisor_1 - 1 NEW_LINE sum_divisor_2 = 1 NEW_LINE j = 2 NEW_LINE while j * j <= number_2 : NEW_LINE INDENT if ( number_2 % j == 0 ) : NEW_LINE INDENT sum_divisor_2 += j NEW_LINE if ( j * j != number_2 ) : NEW_LINE INDENT sum_divisor_2 += number_2 // j NEW_LINE DEDENT DEDENT j = j + 1 NEW_LINE DEDENT if ( sum_divisor_2 == number_1 + 1 and number_1 <= n and number_2 <= n ) : NEW_LINE INDENT Set . append ( number_1 ) NEW_LINE Set . append ( number_2 ) NEW_LINE DEDENT DEDENT
Summ = 0 NEW_LINE for i in Set : NEW_LINE INDENT if i <= n : NEW_LINE INDENT Summ += i NEW_LINE DEDENT DEDENT return Summ NEW_LINE
n = 78 NEW_LINE print ( Betrothed_Sum ( n ) ) NEW_LINE
def rainDayProbability ( a , n ) : NEW_LINE
count = a . count ( 1 ) NEW_LINE
m = count / n NEW_LINE return m NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = [ 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 ] NEW_LINE n = len ( a ) NEW_LINE print ( rainDayProbability ( a , n ) ) NEW_LINE DEDENT
def Series ( n ) : NEW_LINE INDENT sums = 0.0 NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE INDENT ser = 1 / ( i ** i ) NEW_LINE sums += ser NEW_LINE DEDENT return sums NEW_LINE DEDENT
n = 3 NEW_LINE res = round ( Series ( n ) , 5 ) NEW_LINE print ( res ) NEW_LINE
def lexicographicallyMaximum ( S , N ) : NEW_LINE
M = { } NEW_LINE
for i in range ( N ) : NEW_LINE INDENT if S [ i ] in M : NEW_LINE M [ S [ i ] ] += 1 NEW_LINE else : NEW_LINE INDENT M [ S [ i ] ] = 1 NEW_LINE DEDENT DEDENT
V = [ ] NEW_LINE for i in range ( ord ( ' a ' ) , ord ( ' a ' ) + min ( N , 25 ) ) : NEW_LINE INDENT if i not in M : NEW_LINE INDENT V . append ( chr ( i ) ) NEW_LINE DEDENT DEDENT
j = len ( V ) - 1 NEW_LINE
for i in range ( N ) : NEW_LINE
if ( ord ( S [ i ] ) >= ( ord ( ' a ' ) + min ( N , 25 ) ) or ( S [ i ] in M and M [ S [ i ] ] > 1 ) ) : NEW_LINE INDENT if ( ord ( V [ j ] ) < ord ( S [ i ] ) ) : NEW_LINE INDENT continue NEW_LINE DEDENT DEDENT
M [ S [ i ] ] -= 1 NEW_LINE
S = S [ 0 : i ] + V [ j ] + S [ ( i + 1 ) : ] NEW_LINE
j -= 1 NEW_LINE if ( j < 0 ) : NEW_LINE break NEW_LINE l = 0 NEW_LINE
for i in range ( N - 1 , - 1 , - 1 ) : NEW_LINE INDENT if ( l > j ) : NEW_LINE INDENT break NEW_LINE DEDENT if ( ord ( S [ i ] ) >= ( ord ( ' a ' ) + min ( N , 25 ) ) or S [ i ] in M and M [ S [ i ] ] > 1 ) : NEW_LINE DEDENT
M [ S [ i ] ] -= 1 NEW_LINE
S = S [ 0 : i ] + V [ l ] + S [ ( i + 1 ) : ] NEW_LINE
l += 1 NEW_LINE s = list ( S ) NEW_LINE s [ len ( s ) - 1 ] = ' d ' NEW_LINE S = " " . join ( s ) NEW_LINE
return S NEW_LINE
S = " abccefghh " NEW_LINE N = len ( S ) NEW_LINE
print ( lexicographicallyMaximum ( S , N ) ) NEW_LINE
def isConsistingSubarrayUtil ( arr , n ) : NEW_LINE
mp = { } ; NEW_LINE
for i in range ( n ) : NEW_LINE
if arr [ i ] in mp : NEW_LINE INDENT mp [ arr [ i ] ] += 1 ; NEW_LINE DEDENT else : NEW_LINE INDENT mp [ arr [ i ] ] = 1 ; NEW_LINE DEDENT
for it in mp : NEW_LINE
if ( mp [ it ] > 1 ) : NEW_LINE INDENT return True ; NEW_LINE DEDENT
return False ; NEW_LINE
def isConsistingSubarray ( arr , N ) : NEW_LINE INDENT if ( isConsistingSubarrayUtil ( arr , N ) ) : NEW_LINE INDENT print ( " Yes " ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) ; NEW_LINE DEDENT DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE
arr = [ 1 , 2 , 3 , 4 , 5 , 1 ] ; NEW_LINE
N = len ( arr ) ; NEW_LINE
isConsistingSubarray ( arr , N ) ; NEW_LINE
import math NEW_LINE
def createhashmap ( Max ) : NEW_LINE
hashmap = { " " } NEW_LINE
curr = 1 NEW_LINE
prev = 0 NEW_LINE
hashmap . add ( prev ) NEW_LINE
while ( curr <= Max ) : NEW_LINE
hashmap . add ( curr ) NEW_LINE
temp = curr NEW_LINE
curr = curr + prev NEW_LINE
prev = temp NEW_LINE return hashmap NEW_LINE
def SieveOfEratosthenes ( Max ) : NEW_LINE
isPrime = [ 1 for x in range ( Max + 1 ) ] NEW_LINE isPrime [ 0 ] = 0 NEW_LINE isPrime [ 1 ] = 0 NEW_LINE
for p in range ( 0 , int ( math . sqrt ( Max ) ) ) : NEW_LINE
if ( isPrime [ p ] ) : NEW_LINE
for i in range ( 2 * p , Max , p ) : NEW_LINE INDENT isPrime [ i ] = 0 NEW_LINE DEDENT return isPrime NEW_LINE
def cntFibonacciPrime ( arr , N ) : NEW_LINE
Max = arr [ 0 ] NEW_LINE
for i in range ( 0 , N ) : NEW_LINE
Max = max ( Max , arr [ i ] ) NEW_LINE
isPrime = SieveOfEratosthenes ( Max ) NEW_LINE
hashmap = createhashmap ( Max ) NEW_LINE
for i in range ( 0 , N ) : NEW_LINE
if arr [ i ] == 1 : NEW_LINE INDENT continue NEW_LINE DEDENT
if ( ( arr [ i ] in hashmap ) and ( not ( isPrime [ arr [ i ] ] ) ) ) : NEW_LINE
print ( arr [ i ] , end = " ▁ " ) NEW_LINE
arr = [ 13 , 55 , 7 , 3 , 5 , 21 , 233 , 144 , 89 ] NEW_LINE N = len ( arr ) NEW_LINE cntFibonacciPrime ( arr , N ) NEW_LINE
import math NEW_LINE
def key ( N ) : NEW_LINE
num = " " + str ( N ) NEW_LINE ans = 0 NEW_LINE j = 0 NEW_LINE
while j < len ( num ) : NEW_LINE
if ( ( ord ( num [ j ] ) - 48 ) % 2 == 0 ) : NEW_LINE INDENT add = 0 NEW_LINE DEDENT
i = j NEW_LINE while j < len ( num ) : NEW_LINE INDENT add += ord ( num [ j ] ) - 48 NEW_LINE DEDENT
if ( add % 2 == 1 ) : NEW_LINE INDENT break NEW_LINE DEDENT j += 1 NEW_LINE if ( add == 0 ) : NEW_LINE ans *= 10 NEW_LINE else : NEW_LINE digit = int ( math . floor ( math . log10 ( add ) + 1 ) ) NEW_LINE ans *= ( pow ( 10 , digit ) ) NEW_LINE
ans += add NEW_LINE
i = j NEW_LINE else : NEW_LINE
add = 0 NEW_LINE
i = j NEW_LINE while j < len ( num ) : NEW_LINE INDENT add += ord ( num [ j ] ) - 48 NEW_LINE DEDENT
if ( add % 2 == 0 ) : NEW_LINE INDENT break NEW_LINE DEDENT j += 1 NEW_LINE if ( add == 0 ) : NEW_LINE ans *= 10 NEW_LINE else : NEW_LINE digit = int ( math . floor ( math . log10 ( add ) + 1 ) ) NEW_LINE ans *= ( pow ( 10 , digit ) ) NEW_LINE
ans += add NEW_LINE
i = j NEW_LINE j += 1 NEW_LINE
if ( j + 1 ) >= len ( num ) : NEW_LINE INDENT return ans NEW_LINE DEDENT else : NEW_LINE INDENT ans += ord ( num [ len ( num ) - 1 ] ) - 48 NEW_LINE return ans NEW_LINE DEDENT
/ * Driver Code * / NEW_LINE N = 1667848271 NEW_LINE print ( key ( N ) ) NEW_LINE
def sentinelSearch ( arr , n , key ) : NEW_LINE
def sentinelSearch ( arr , n , key ) : NEW_LINE
last = arr [ n - 1 ] NEW_LINE
arr [ n - 1 ] = key NEW_LINE i = 0 NEW_LINE while ( arr [ i ] != key ) : NEW_LINE INDENT i += 1 NEW_LINE DEDENT
arr [ n - 1 ] = last NEW_LINE if ( ( i < n - 1 ) or ( arr [ n - 1 ] == key ) ) : NEW_LINE INDENT print ( key , " is ▁ present ▁ at ▁ index " , i ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Element ▁ Not ▁ found " ) NEW_LINE DEDENT
arr = [ 10 , 20 , 180 , 30 , 60 , 50 , 110 , 100 , 70 ] NEW_LINE n = len ( arr ) NEW_LINE key = 180 NEW_LINE sentinelSearch ( arr , n , key ) NEW_LINE
def maximum_middle_value ( n , k , arr ) : NEW_LINE
ans = - 1 NEW_LINE
low = ( n + 1 - k ) // 2 NEW_LINE high = ( n + 1 - k ) // 2 + k NEW_LINE
for i in range ( low , high + 1 ) : NEW_LINE
ans = max ( ans , arr [ i - 1 ] ) NEW_LINE
return ans NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n , k = 5 , 2 NEW_LINE arr = [ 9 , 5 , 3 , 7 , 10 ] NEW_LINE print ( maximum_middle_value ( n , k , arr ) ) NEW_LINE n , k = 9 , 3 NEW_LINE arr1 = [ 2 , 4 , 3 , 9 , 5 , 8 , 7 , 6 , 10 ] NEW_LINE print ( maximum_middle_value ( n , k , arr1 ) ) NEW_LINE DEDENT
import math as mt NEW_LINE
def ternarySearch ( l , r , key , ar ) : NEW_LINE INDENT if ( r >= l ) : NEW_LINE DEDENT
mid1 = l + ( r - l ) // 3 NEW_LINE mid2 = r - ( r - l ) // 3 NEW_LINE
if ( ar [ mid1 ] == key ) : NEW_LINE INDENT return mid1 NEW_LINE DEDENT if ( ar [ mid2 ] == key ) : NEW_LINE INDENT return mid2 NEW_LINE DEDENT
if ( key < ar [ mid1 ] ) : NEW_LINE
return ternarySearch ( l , mid1 - 1 , key , ar ) NEW_LINE elif ( key > ar [ mid2 ] ) : NEW_LINE
return ternarySearch ( mid2 + 1 , r , key , ar ) NEW_LINE else : NEW_LINE
return ternarySearch ( mid1 + 1 , mid2 - 1 , key , ar ) NEW_LINE
return - 1 NEW_LINE
l , r , p = 0 , 9 , 5 NEW_LINE
ar = [ 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 ] NEW_LINE
l = 0 NEW_LINE
r = 9 NEW_LINE
key = 5 NEW_LINE
p = ternarySearch ( l , r , key , ar ) NEW_LINE
print ( " Index ▁ of " , key , " is " , p ) NEW_LINE
key = 50 NEW_LINE
p = ternarySearch ( l , r , key , ar ) NEW_LINE
print ( " Index ▁ of " , key , " is " , p ) NEW_LINE
def findmin ( p , n ) : NEW_LINE INDENT a , b , c , d = 0 , 0 , 0 , 0 NEW_LINE for i in range ( n ) : NEW_LINE DEDENT
if ( p [ i ] [ 0 ] <= 0 ) : NEW_LINE INDENT a += 1 NEW_LINE DEDENT
elif ( p [ i ] [ 0 ] >= 0 ) : NEW_LINE INDENT b += 1 NEW_LINE DEDENT
if ( p [ i ] [ 1 ] >= 0 ) : NEW_LINE INDENT c += 1 NEW_LINE DEDENT
elif ( p [ i ] [ 1 ] <= 0 ) : NEW_LINE INDENT d += 1 NEW_LINE DEDENT return min ( [ a , b , c , d ] ) NEW_LINE
p = [ [ 1 , 1 ] , [ 2 , 2 ] , [ - 1 , - 1 ] , [ - 2 , 2 ] ] NEW_LINE n = len ( p ) NEW_LINE print ( findmin ( p , n ) ) NEW_LINE
def maxOps ( a , b , c ) : NEW_LINE
INDENT arr = [ a , b , c ] NEW_LINE DEDENT
INDENT count = 0 NEW_LINE while True : NEW_LINE DEDENT
arr . sort ( ) NEW_LINE
if not arr [ 0 ] and not arr [ 1 ] : NEW_LINE break NEW_LINE
arr [ 1 ] -= 1 NEW_LINE arr [ 2 ] -= 1 NEW_LINE
count += 1 NEW_LINE
INDENT print ( count ) NEW_LINE DEDENT
a , b , c = 4 , 3 , 2 NEW_LINE maxOps ( a , b , c ) NEW_LINE
MAX = 26 NEW_LINE
lower = [ 0 ] * MAX ; NEW_LINE upper = [ 0 ] * MAX ; NEW_LINE for i in range ( n ) : NEW_LINE
if ( s [ i ] . islower ( ) ) : NEW_LINE INDENT lower [ ord ( s [ i ] ) - ord ( ' a ' ) ] += 1 ; NEW_LINE DEDENT
elif ( s [ i ] . isupper ( ) ) : NEW_LINE INDENT upper [ ord ( s [ i ] ) - ord ( ' A ' ) ] += 1 ; NEW_LINE DEDENT
i = 0 ; j = 0 ; NEW_LINE while ( i < MAX and lower [ i ] == 0 ) : NEW_LINE INDENT i += 1 ; NEW_LINE DEDENT while ( j < MAX and upper [ j ] == 0 ) : NEW_LINE INDENT j += 1 ; NEW_LINE DEDENT
for k in range ( n ) : NEW_LINE
if ( s [ k ] . islower ( ) ) : NEW_LINE INDENT while ( lower [ i ] == 0 ) : NEW_LINE INDENT i += 1 ; NEW_LINE DEDENT s [ k ] = chr ( i + ord ( ' a ' ) ) ; NEW_LINE DEDENT
lower [ i ] -= 1 ; NEW_LINE
elif ( s [ k ] . isupper ( ) ) : NEW_LINE INDENT while ( upper [ j ] == 0 ) : NEW_LINE INDENT j += 1 ; NEW_LINE DEDENT s [ k ] = chr ( j + ord ( ' A ' ) ) ; NEW_LINE DEDENT
upper [ j ] -= 1 ; NEW_LINE
return " " . join ( s ) ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT s = " gEeksfOrgEEkS " ; NEW_LINE n = len ( s ) ; NEW_LINE print ( getSortedString ( list ( s ) , n ) ) ; NEW_LINE DEDENT
import numpy as np NEW_LINE
def prCharWithFreq ( str ) : NEW_LINE
' NEW_LINE INDENT n = len ( str ) NEW_LINE DEDENT
' NEW_LINE INDENT for i in range ( 0 , n ) : NEW_LINE INDENT freq [ ord ( str [ i ] ) - ord ( ' a ' ) ] += 1 NEW_LINE DEDENT DEDENT
for i in range ( 0 , n ) : NEW_LINE
if ( freq [ ord ( str [ i ] ) - ord ( ' a ' ) ] != 0 ) : NEW_LINE
print ( str [ i ] , freq [ ord ( str [ i ] ) - ord ( ' a ' ) ] , end = " ▁ " ) NEW_LINE
freq [ ord ( str [ i ] ) - ord ( ' a ' ) ] = 0 NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT str = " geeksforgeeks " ; NEW_LINE prCharWithFreq ( str ) ; NEW_LINE DEDENT ' NEW_LINE
s = " i ▁ like ▁ this ▁ program ▁ very ▁ much " NEW_LINE words = s . split ( ' ▁ ' ) NEW_LINE string = [ ] NEW_LINE for word in words : NEW_LINE INDENT string . insert ( 0 , word ) NEW_LINE DEDENT print ( " Reversed ▁ String : " ) NEW_LINE print ( " ▁ " . join ( string ) ) NEW_LINE
def SieveOfEratosthenes ( prime , n ) : NEW_LINE INDENT p = 2 NEW_LINE while ( p * p <= n ) : NEW_LINE DEDENT
if ( prime [ p ] == True ) : NEW_LINE
i = p * p NEW_LINE while ( i <= n ) : NEW_LINE INDENT prime [ i ] = False NEW_LINE i += p NEW_LINE DEDENT p += 1 NEW_LINE
def segregatePrimeNonPrime ( prime , arr , N ) : NEW_LINE
SieveOfEratosthenes ( prime , 10000000 ) NEW_LINE
left , right = 0 , N - 1 NEW_LINE
while ( left < right ) : NEW_LINE
while ( prime [ arr [ left ] ] ) : NEW_LINE INDENT left += 1 NEW_LINE DEDENT
while ( not prime [ arr [ right ] ] ) : NEW_LINE INDENT right -= 1 NEW_LINE DEDENT
if ( left < right ) : NEW_LINE
arr [ left ] , arr [ right ] = arr [ right ] , arr [ left ] NEW_LINE left += 1 NEW_LINE right -= 1 NEW_LINE
for num in arr : NEW_LINE INDENT print ( num , end = " ▁ " ) NEW_LINE DEDENT
arr = [ 2 , 3 , 4 , 6 , 7 , 8 , 9 , 10 ] NEW_LINE N = len ( arr ) NEW_LINE prime = [ True ] * 10000001 NEW_LINE
segregatePrimeNonPrime ( prime , arr , N ) NEW_LINE
def findDepthRec ( tree , n , index ) : NEW_LINE INDENT if ( index [ 0 ] >= n or tree [ index [ 0 ] ] == ' l ' ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT DEDENT
index [ 0 ] += 1 NEW_LINE left = findDepthRec ( tree , n , index ) NEW_LINE
index [ 0 ] += 1 NEW_LINE right = findDepthRec ( tree , n , index ) NEW_LINE return ( max ( left , right ) + 1 ) NEW_LINE
def findDepth ( tree , n ) : NEW_LINE INDENT index = [ 0 ] NEW_LINE return findDepthRec ( tree , n , index ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT tree = " nlnnlll " NEW_LINE n = len ( tree ) NEW_LINE print ( findDepth ( tree , n ) ) NEW_LINE DEDENT
class newNode : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . key = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def insert ( node , key ) : NEW_LINE
if node == None : NEW_LINE INDENT return newNode ( key ) NEW_LINE DEDENT
if key < node . key : NEW_LINE INDENT node . left = insert ( node . left , key ) NEW_LINE DEDENT elif key > node . key : NEW_LINE INDENT node . right = insert ( node . right , key ) NEW_LINE DEDENT
return node NEW_LINE
def findMaxforN ( root , N ) : NEW_LINE
if root == None : NEW_LINE INDENT return - 1 NEW_LINE DEDENT if root . key == N : NEW_LINE INDENT return N NEW_LINE DEDENT
elif root . key < N : NEW_LINE INDENT k = findMaxforN ( root . right , N ) NEW_LINE if k == - 1 : NEW_LINE INDENT return root . key NEW_LINE DEDENT else : NEW_LINE INDENT return k NEW_LINE DEDENT DEDENT
elif root . key > N : NEW_LINE INDENT return findMaxforN ( root . left , N ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 4 NEW_LINE DEDENT
root = None NEW_LINE root = insert ( root , 25 ) NEW_LINE insert ( root , 2 ) NEW_LINE insert ( root , 1 ) NEW_LINE insert ( root , 3 ) NEW_LINE insert ( root , 12 ) NEW_LINE insert ( root , 9 ) NEW_LINE insert ( root , 21 ) NEW_LINE insert ( root , 19 ) NEW_LINE insert ( root , 25 ) NEW_LINE print ( findMaxforN ( root , N ) ) NEW_LINE
class createNode : NEW_LINE
def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT
def insertNode ( root , x ) : NEW_LINE INDENT p , q = root , None NEW_LINE while p != None : NEW_LINE INDENT q = p NEW_LINE if p . data < x : NEW_LINE INDENT p = p . right NEW_LINE DEDENT else : NEW_LINE INDENT p = p . left NEW_LINE DEDENT DEDENT if q == None : NEW_LINE INDENT p = createNode ( x ) NEW_LINE DEDENT else : NEW_LINE INDENT if q . data < x : NEW_LINE INDENT q . right = createNode ( x ) NEW_LINE DEDENT else : NEW_LINE INDENT q . left = createNode ( x ) NEW_LINE DEDENT DEDENT DEDENT
def maxelpath ( q , x ) : NEW_LINE INDENT p = q NEW_LINE mx = - 999999999999 NEW_LINE DEDENT
while p . data != x : NEW_LINE INDENT if p . data > x : NEW_LINE INDENT mx = max ( mx , p . data ) NEW_LINE p = p . left NEW_LINE DEDENT else : NEW_LINE INDENT mx = max ( mx , p . data ) NEW_LINE p = p . right NEW_LINE DEDENT DEDENT return max ( mx , x ) NEW_LINE
def maximumElement ( root , x , y ) : NEW_LINE INDENT p = root NEW_LINE DEDENT
while ( ( x < p . data and y < p . data ) or ( x > p . data and y > p . data ) ) : NEW_LINE
if x < p . data and y < p . data : NEW_LINE INDENT p = p . left NEW_LINE DEDENT
elif x > p . data and y > p . data : NEW_LINE INDENT p = p . right NEW_LINE DEDENT
return max ( maxelpath ( p , x ) , maxelpath ( p , y ) ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 18 , 36 , 9 , 6 , 12 , 10 , 1 , 8 ] NEW_LINE a , b = 1 , 10 NEW_LINE n = len ( arr ) NEW_LINE DEDENT
root = createNode ( arr [ 0 ] ) NEW_LINE
for i in range ( 1 , n ) : NEW_LINE INDENT insertNode ( root , arr [ i ] ) NEW_LINE DEDENT print ( maximumElement ( root , a , b ) ) NEW_LINE
class newNode : NEW_LINE INDENT def __init__ ( self , key ) : NEW_LINE DEDENT
self . info = key NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE self . lthread = True NEW_LINE
self . rthread = True NEW_LINE
def insert ( root , ikey ) : NEW_LINE
ptr = root NEW_LINE
par = None NEW_LINE while ptr != None : NEW_LINE
if ikey == ( ptr . info ) : NEW_LINE INDENT print ( " Duplicate ▁ Key ▁ ! " ) NEW_LINE return root NEW_LINE DEDENT
par = ptr NEW_LINE
if ikey < ptr . info : NEW_LINE INDENT if ptr . lthread == False : NEW_LINE INDENT ptr = ptr . left NEW_LINE DEDENT else : NEW_LINE INDENT break NEW_LINE DEDENT DEDENT
else : NEW_LINE INDENT if ptr . rthread == False : NEW_LINE INDENT ptr = ptr . right NEW_LINE DEDENT else : NEW_LINE INDENT break NEW_LINE DEDENT DEDENT
tmp = newNode ( ikey ) NEW_LINE if par == None : NEW_LINE INDENT root = tmp NEW_LINE tmp . left = None NEW_LINE tmp . right = None NEW_LINE DEDENT elif ikey < ( par . info ) : NEW_LINE INDENT tmp . left = par . left NEW_LINE tmp . right = par NEW_LINE par . lthread = False NEW_LINE par . left = tmp NEW_LINE DEDENT else : NEW_LINE INDENT tmp . left = par NEW_LINE tmp . right = par . right NEW_LINE par . rthread = False NEW_LINE par . right = tmp NEW_LINE DEDENT return root NEW_LINE
def inorderSuccessor ( ptr ) : NEW_LINE
if ptr . rthread == True : NEW_LINE INDENT return ptr . right NEW_LINE DEDENT
ptr = ptr . right NEW_LINE while ptr . lthread == False : NEW_LINE INDENT ptr = ptr . left NEW_LINE DEDENT return ptr NEW_LINE
def inorder ( root ) : NEW_LINE INDENT if root == None : NEW_LINE INDENT print ( " Tree ▁ is ▁ empty " ) NEW_LINE DEDENT DEDENT
ptr = root NEW_LINE while ptr . lthread == False : NEW_LINE INDENT ptr = ptr . left NEW_LINE DEDENT
while ptr != None : NEW_LINE INDENT print ( ptr . info , end = " ▁ " ) NEW_LINE ptr = inorderSuccessor ( ptr ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT root = None NEW_LINE root = insert ( root , 20 ) NEW_LINE root = insert ( root , 10 ) NEW_LINE root = insert ( root , 30 ) NEW_LINE root = insert ( root , 5 ) NEW_LINE root = insert ( root , 16 ) NEW_LINE root = insert ( root , 14 ) NEW_LINE root = insert ( root , 17 ) NEW_LINE root = insert ( root , 13 ) NEW_LINE inorder ( root ) NEW_LINE DEDENT
MAX = 1000 NEW_LINE def checkHV ( arr , N , M ) : NEW_LINE
horizontal = True NEW_LINE vertical = True NEW_LINE
i = 0 NEW_LINE k = N - 1 NEW_LINE while ( i < N // 2 ) : NEW_LINE
for j in range ( M ) : NEW_LINE
if ( arr [ i ] [ j ] != arr [ k ] [ j ] ) : NEW_LINE INDENT horizontal = False NEW_LINE break NEW_LINE DEDENT i += 1 NEW_LINE k -= 1 NEW_LINE
i = 0 NEW_LINE k = M - 1 NEW_LINE while ( i < M // 2 ) : NEW_LINE
for j in range ( N ) : NEW_LINE
if ( arr [ i ] [ j ] != arr [ k ] [ j ] ) : NEW_LINE INDENT vertical = False NEW_LINE break NEW_LINE DEDENT i += 1 NEW_LINE k -= 1 NEW_LINE if ( not horizontal and not vertical ) : NEW_LINE print ( " NO " ) NEW_LINE elif ( horizontal and not vertical ) : NEW_LINE print ( " HORIZONTAL " ) NEW_LINE elif ( vertical and not horizontal ) : NEW_LINE print ( " VERTICAL " ) NEW_LINE else : NEW_LINE print ( " BOTH " ) NEW_LINE
mat = [ [ 1 , 0 , 1 ] , [ 0 , 0 , 0 ] , [ 1 , 0 , 1 ] ] NEW_LINE checkHV ( mat , 3 , 3 ) NEW_LINE
R = 3 NEW_LINE C = 4 NEW_LINE
def gcd ( a , b ) : NEW_LINE INDENT if ( b == 0 ) : NEW_LINE INDENT return a NEW_LINE DEDENT return gcd ( b , a % b ) NEW_LINE DEDENT
def replacematrix ( mat , n , m ) : NEW_LINE INDENT rgcd = [ 0 ] * R NEW_LINE cgcd = [ 0 ] * C NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT for j in range ( m ) : NEW_LINE INDENT rgcd [ i ] = gcd ( rgcd [ i ] , mat [ i ] [ j ] ) NEW_LINE cgcd [ j ] = gcd ( cgcd [ j ] , mat [ i ] [ j ] ) NEW_LINE DEDENT DEDENT
for i in range ( n ) : NEW_LINE INDENT for j in range ( m ) : NEW_LINE INDENT mat [ i ] [ j ] = max ( rgcd [ i ] , cgcd [ j ] ) NEW_LINE DEDENT DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT m = [ [ 1 , 2 , 3 , 3 ] , [ 4 , 5 , 6 , 6 ] , [ 7 , 8 , 9 , 9 ] ] NEW_LINE replacematrix ( m , R , C ) NEW_LINE for i in range ( R ) : NEW_LINE INDENT for j in range ( C ) : NEW_LINE INDENT print ( m [ i ] [ j ] , end = " ▁ " ) NEW_LINE DEDENT print ( ) NEW_LINE DEDENT DEDENT
N = 4 NEW_LINE
def add ( A , B , C ) : NEW_LINE INDENT for i in range ( N ) : NEW_LINE INDENT for j in range ( N ) : NEW_LINE INDENT C [ i ] [ j ] = A [ i ] [ j ] + B [ i ] [ j ] NEW_LINE DEDENT DEDENT DEDENT
A = [ [ 1 , 1 , 1 , 1 ] , [ 2 , 2 , 2 , 2 ] , [ 3 , 3 , 3 , 3 ] , [ 4 , 4 , 4 , 4 ] ] NEW_LINE B = [ [ 1 , 1 , 1 , 1 ] , [ 2 , 2 , 2 , 2 ] , [ 3 , 3 , 3 , 3 ] , [ 4 , 4 , 4 , 4 ] ] NEW_LINE C = A [ : ] [ : ] NEW_LINE add ( A , B , C ) NEW_LINE print ( " Result ▁ matrix ▁ is " ) NEW_LINE for i in range ( N ) : NEW_LINE INDENT for j in range ( N ) : NEW_LINE INDENT print ( C [ i ] [ j ] , " ▁ " , end = ' ' ) NEW_LINE DEDENT print ( ) NEW_LINE DEDENT
N = 4 NEW_LINE
def subtract ( A , B , C ) : NEW_LINE INDENT for i in range ( N ) : NEW_LINE INDENT for j in range ( N ) : NEW_LINE INDENT C [ i ] [ j ] = A [ i ] [ j ] - B [ i ] [ j ] NEW_LINE DEDENT DEDENT DEDENT
A = [ [ 1 , 1 , 1 , 1 ] , [ 2 , 2 , 2 , 2 ] , [ 3 , 3 , 3 , 3 ] , [ 4 , 4 , 4 , 4 ] ] NEW_LINE B = [ [ 1 , 1 , 1 , 1 ] , [ 2 , 2 , 2 , 2 ] , [ 3 , 3 , 3 , 3 ] , [ 4 , 4 , 4 , 4 ] ] NEW_LINE C = A [ : ] [ : ] NEW_LINE subtract ( A , B , C ) NEW_LINE print ( " Result ▁ matrix ▁ is " ) NEW_LINE for i in range ( N ) : NEW_LINE INDENT for j in range ( N ) : NEW_LINE INDENT print ( C [ i ] [ j ] , " ▁ " , end = ' ' ) NEW_LINE DEDENT print ( ) NEW_LINE DEDENT
def linearSearch ( arr , n ) : NEW_LINE INDENT for i in range ( n ) : NEW_LINE INDENT if arr [ i ] is i : NEW_LINE INDENT return i NEW_LINE DEDENT DEDENT DEDENT
return - 1 NEW_LINE
arr = [ - 10 , - 1 , 0 , 3 , 10 , 11 , 30 , 50 , 100 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Fixed ▁ Point ▁ is ▁ " + str ( linearSearch ( arr , n ) ) ) NEW_LINE
def binarySearch ( arr , low , high ) : NEW_LINE INDENT if high >= low : NEW_LINE DEDENT
mid = ( low + high ) // 2 NEW_LINE if mid is arr [ mid ] : NEW_LINE return mid NEW_LINE if mid > arr [ mid ] : NEW_LINE return binarySearch ( arr , ( mid + 1 ) , high ) NEW_LINE else : NEW_LINE return binarySearch ( arr , low , ( mid - 1 ) ) NEW_LINE
return - 1 NEW_LINE
arr = [ - 10 , - 1 , 0 , 3 , 10 , 11 , 30 , 50 , 100 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Fixed ▁ Point ▁ is ▁ " + str ( binarySearch ( arr , 0 , n - 1 ) ) ) NEW_LINE
def maxTripletSum ( arr , n ) : NEW_LINE
sm = - 1000000 NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT for j in range ( i + 1 , n ) : NEW_LINE INDENT for k in range ( j + 1 , n ) : NEW_LINE INDENT if ( sm < ( arr [ i ] + arr [ j ] + arr [ k ] ) ) : NEW_LINE INDENT sm = arr [ i ] + arr [ j ] + arr [ k ] NEW_LINE DEDENT DEDENT DEDENT DEDENT return sm NEW_LINE
arr = [ 1 , 0 , 8 , 6 , 4 , 2 ] NEW_LINE n = len ( arr ) NEW_LINE print ( maxTripletSum ( arr , n ) ) NEW_LINE
def maxTripletSum ( arr , n ) : NEW_LINE
arr . sort ( ) NEW_LINE
return ( arr [ n - 1 ] + arr [ n - 2 ] + arr [ n - 3 ] ) NEW_LINE
arr = [ 1 , 0 , 8 , 6 , 4 , 2 ] NEW_LINE n = len ( arr ) NEW_LINE print ( maxTripletSum ( arr , n ) ) NEW_LINE
def maxTripletSum ( arr , n ) : NEW_LINE
maxA = - 100000000 NEW_LINE maxB = - 100000000 NEW_LINE maxC = - 100000000 NEW_LINE for i in range ( 0 , n ) : NEW_LINE
if ( arr [ i ] > maxA ) : NEW_LINE INDENT maxC = maxB NEW_LINE maxB = maxA NEW_LINE maxA = arr [ i ] NEW_LINE DEDENT
elif ( arr [ i ] > maxB ) : NEW_LINE INDENT maxC = maxB NEW_LINE maxB = arr [ i ] NEW_LINE DEDENT
elif ( arr [ i ] > maxC ) : NEW_LINE INDENT maxC = arr [ i ] NEW_LINE DEDENT return ( maxA + maxB + maxC ) NEW_LINE
arr = [ 1 , 0 , 8 , 6 , 4 , 2 ] NEW_LINE n = len ( arr ) NEW_LINE print ( maxTripletSum ( arr , n ) ) NEW_LINE
def search ( arr , n , x ) : NEW_LINE INDENT for i in range ( 0 , n ) : NEW_LINE INDENT if ( arr [ i ] == x ) : NEW_LINE INDENT return i NEW_LINE DEDENT DEDENT return - 1 NEW_LINE DEDENT
arr = [ 2 , 3 , 4 , 10 , 40 ] NEW_LINE x = 10 NEW_LINE n = len ( arr ) NEW_LINE
result = search ( arr , n , x ) NEW_LINE if ( result == - 1 ) : NEW_LINE INDENT print ( " Element ▁ is ▁ not ▁ present ▁ in ▁ array " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Element ▁ is ▁ present ▁ at ▁ index " , result ) NEW_LINE DEDENT
def search ( arr , search_Element ) : NEW_LINE INDENT left = 0 NEW_LINE length = len ( arr ) NEW_LINE position = - 1 NEW_LINE right = length - 1 NEW_LINE DEDENT
for left in range ( 0 , right , 1 ) : NEW_LINE
if ( arr [ left ] == search_Element ) : NEW_LINE INDENT position = left NEW_LINE print ( " Element ▁ found ▁ in ▁ Array ▁ at ▁ " , position + 1 , " ▁ Position ▁ with ▁ " , left + 1 , " ▁ Attempt " ) NEW_LINE break NEW_LINE DEDENT
if ( arr [ right ] == search_Element ) : NEW_LINE INDENT position = right NEW_LINE print ( " Element ▁ found ▁ in ▁ Array ▁ at ▁ " , position + 1 , " ▁ Position ▁ with ▁ " , length - right , " ▁ Attempt " ) NEW_LINE break NEW_LINE DEDENT left += 1 NEW_LINE right -= 1 NEW_LINE
if ( position == - 1 ) : NEW_LINE INDENT print ( " Not ▁ found ▁ in ▁ Array ▁ with ▁ " , left , " ▁ Attempt " ) NEW_LINE DEDENT
arr = [ 1 , 2 , 3 , 4 , 5 ] NEW_LINE search_element = 5 NEW_LINE
search ( arr , search_element ) NEW_LINE
def countSort ( arr ) : NEW_LINE
output = [ 0 for i in range ( len ( arr ) ) ] NEW_LINE
count = [ 0 for i in range ( 256 ) ] NEW_LINE
for i in arr : NEW_LINE INDENT count [ ord ( i ) ] += 1 NEW_LINE DEDENT
for i in range ( 256 ) : NEW_LINE INDENT count [ i ] += count [ i - 1 ] NEW_LINE DEDENT
for i in range ( len ( arr ) ) : NEW_LINE INDENT output [ count [ ord ( arr [ i ] ) ] - 1 ] = arr [ i ] NEW_LINE count [ ord ( arr [ i ] ) ] -= 1 NEW_LINE DEDENT
ans = [ " " for _ in arr ] NEW_LINE for i in range ( len ( arr ) ) : NEW_LINE INDENT ans [ i ] = output [ i ] NEW_LINE DEDENT return ans NEW_LINE
arr = " geeksforgeeks " NEW_LINE ans = countSort ( arr ) NEW_LINE print ( " Sorted ▁ character ▁ array ▁ is ▁ % ▁ s " % ( " " . join ( ans ) ) ) NEW_LINE
def count_sort ( arr ) : NEW_LINE INDENT max_element = int ( max ( arr ) ) NEW_LINE min_element = int ( min ( arr ) ) NEW_LINE range_of_elements = max_element - min_element + 1 NEW_LINE count_arr = [ 0 for _ in range ( range_of_elements ) ] NEW_LINE output_arr = [ 0 for _ in range ( len ( arr ) ) ] NEW_LINE for i in range ( 0 , len ( arr ) ) : NEW_LINE INDENT count_arr [ arr [ i ] - min_element ] += 1 NEW_LINE DEDENT for i in range ( 1 , len ( count_arr ) ) : NEW_LINE INDENT count_arr [ i ] += count_arr [ i - 1 ] NEW_LINE DEDENT for i in range ( len ( arr ) - 1 , - 1 , - 1 ) : NEW_LINE INDENT output_arr [ count_arr [ arr [ i ] - min_element ] - 1 ] = arr [ i ] NEW_LINE count_arr [ arr [ i ] - min_element ] -= 1 NEW_LINE DEDENT for i in range ( 0 , len ( arr ) ) : NEW_LINE INDENT arr [ i ] = output_arr [ i ] NEW_LINE DEDENT return arr NEW_LINE DEDENT
arr = [ - 5 , - 10 , 0 , - 3 , 8 , 5 , - 1 , 10 ] NEW_LINE ans = count_sort ( arr ) NEW_LINE print ( " Sorted ▁ character ▁ array ▁ is ▁ " + str ( ans ) ) NEW_LINE
def binomialCoeff ( n , k ) : NEW_LINE
if k > n : NEW_LINE INDENT return 0 NEW_LINE DEDENT if k == 0 or k == n : NEW_LINE INDENT return 1 NEW_LINE DEDENT
return binomialCoeff ( n - 1 , k - 1 ) + binomialCoeff ( n - 1 , k ) NEW_LINE
n = 5 NEW_LINE k = 2 NEW_LINE print " Value ▁ of ▁ C ( % d , % d ) ▁ is ▁ ( % d ) " % ( n , k , binomialCoeff ( n , k ) ) NEW_LINE
def binomialCoeff ( n , k ) : NEW_LINE INDENT C = [ 0 for i in xrange ( k + 1 ) ] NEW_LINE DEDENT
C [ 0 ] = 1 NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE
j = min ( i , k ) NEW_LINE while ( j > 0 ) : NEW_LINE INDENT C [ j ] = C [ j ] + C [ j - 1 ] NEW_LINE j -= 1 NEW_LINE DEDENT return C [ k ] NEW_LINE
n = 5 NEW_LINE k = 2 NEW_LINE print " Value ▁ of ▁ C ( % d , % d ) ▁ is ▁ % d " % ( n , k , binomialCoeff ( n , k ) ) NEW_LINE
import math NEW_LINE class GFG : NEW_LINE
def nCr ( self , n , r ) : NEW_LINE INDENT if r > n : NEW_LINE INDENT return 0 NEW_LINE DEDENT DEDENT
if n - r > r : NEW_LINE INDENT r = n - r NEW_LINE DEDENT
SPF = [ i for i in range ( n + 1 ) ] NEW_LINE
for i in range ( 4 , n + 1 , 2 ) : NEW_LINE INDENT SPF [ i ] = 2 NEW_LINE DEDENT for i in range ( 3 , n + 1 , 2 ) : NEW_LINE INDENT if i * i > n : NEW_LINE INDENT break NEW_LINE DEDENT DEDENT
if SPF [ i ] == i : NEW_LINE
for j in range ( i * i , n + 1 , i ) : NEW_LINE INDENT if SPF [ j ] == j : NEW_LINE INDENT SPF [ j ] = i NEW_LINE DEDENT DEDENT
prime_pow = { } NEW_LINE
for i in range ( r + 1 , n + 1 ) : NEW_LINE INDENT t = i NEW_LINE DEDENT
while t > 1 : NEW_LINE INDENT if not SPF [ t ] in prime_pow : NEW_LINE INDENT prime_pow [ SPF [ t ] ] = 1 NEW_LINE DEDENT else : NEW_LINE INDENT prime_pow [ SPF [ t ] ] += 1 NEW_LINE DEDENT t //= SPF [ t ] NEW_LINE DEDENT
for i in range ( 1 , n - r + 1 ) : NEW_LINE INDENT t = i NEW_LINE DEDENT
while t > 1 : NEW_LINE INDENT prime_pow [ SPF [ t ] ] -= 1 NEW_LINE t //= SPF [ t ] NEW_LINE DEDENT
ans = 1 NEW_LINE mod = 10 ** 9 + 7 NEW_LINE
for i in prime_pow : NEW_LINE
ans = ( ans * pow ( i , prime_pow [ i ] , mod ) ) % mod NEW_LINE return ans NEW_LINE
n = 5 NEW_LINE k = 2 NEW_LINE ob = GFG ( ) NEW_LINE print ( " Value ▁ of ▁ C ( " + str ( n ) + " , ▁ " + str ( k ) + " ) ▁ is " , ob . nCr ( n , k ) ) NEW_LINE
def binomialCoeff ( n , r ) : NEW_LINE INDENT if ( r > n ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT m = 1000000007 NEW_LINE inv = [ 0 for i in range ( r + 1 ) ] NEW_LINE inv [ 0 ] = 1 NEW_LINE if ( r + 1 >= 2 ) : NEW_LINE INDENT inv [ 1 ] = 1 NEW_LINE DEDENT DEDENT
for i in range ( 2 , r + 1 ) : NEW_LINE INDENT inv [ i ] = m - ( m // i ) * inv [ m % i ] % m NEW_LINE DEDENT ans = 1 NEW_LINE
for i in range ( 2 , r + 1 ) : NEW_LINE INDENT ans = ( ( ans % m ) * ( inv [ i ] % m ) ) % m NEW_LINE DEDENT
for i in range ( n , n - r , - 1 ) : NEW_LINE INDENT ans = ( ( ans % m ) * ( i % m ) ) % m NEW_LINE DEDENT return ans NEW_LINE
n = 5 NEW_LINE r = 2 NEW_LINE print ( " Value ▁ of ▁ C ( " , n , " , ▁ " , r , " ) ▁ is ▁ " , binomialCoeff ( n , r ) ) NEW_LINE
def findPartiion ( arr , n ) : NEW_LINE INDENT Sum = 0 NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT Sum += arr [ i ] NEW_LINE DEDENT if ( Sum % 2 != 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT part = [ 0 ] * ( ( Sum // 2 ) + 1 ) NEW_LINE
for i in range ( ( Sum // 2 ) + 1 ) : NEW_LINE INDENT part [ i ] = 0 NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE
for j in range ( Sum // 2 , arr [ i ] - 1 , - 1 ) : NEW_LINE
if ( part [ j - arr [ i ] ] == 1 or j == arr [ i ] ) : NEW_LINE INDENT part [ j ] = 1 NEW_LINE DEDENT return part [ Sum // 2 ] NEW_LINE
arr = [ 1 , 3 , 3 , 2 , 3 , 2 ] NEW_LINE n = len ( arr ) NEW_LINE
if ( findPartiion ( arr , n ) == 1 ) : NEW_LINE INDENT print ( " Can ▁ be ▁ divided ▁ into ▁ two ▁ subsets ▁ of ▁ equal ▁ sum " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Can ▁ not ▁ be ▁ divided ▁ into ▁ two ▁ subsets ▁ of ▁ equal ▁ sum " ) NEW_LINE DEDENT
def isSubsetSum ( set , n , sum ) : NEW_LINE
if ( sum == 0 ) : NEW_LINE INDENT return True NEW_LINE DEDENT if ( n == 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT
if ( set [ n - 1 ] > sum ) : NEW_LINE INDENT return isSubsetSum ( set , n - 1 , sum ) NEW_LINE DEDENT
return isSubsetSum ( set , n - 1 , sum ) or isSubsetSum ( set , n - 1 , sum - set [ n - 1 ] ) NEW_LINE
set = [ 3 , 34 , 4 , 12 , 5 , 2 ] NEW_LINE sum = 9 NEW_LINE n = len ( set ) NEW_LINE if ( isSubsetSum ( set , n , sum ) == True ) : NEW_LINE INDENT print ( " Found ▁ a ▁ subset ▁ with ▁ given ▁ sum " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No ▁ subset ▁ with ▁ given ▁ sum " ) NEW_LINE DEDENT
def isSubsetSum ( set , n , sum ) : NEW_LINE
subset = ( [ [ False for i in range ( sum + 1 ) ] for i in range ( n + 1 ) ] ) NEW_LINE
for i in range ( n + 1 ) : NEW_LINE INDENT subset [ i ] [ 0 ] = True NEW_LINE DEDENT
for i in range ( 1 , sum + 1 ) : NEW_LINE INDENT subset [ 0 ] [ i ] = False NEW_LINE DEDENT
for i in range ( 1 , n + 1 ) : NEW_LINE INDENT for j in range ( 1 , sum + 1 ) : NEW_LINE INDENT if j < set [ i - 1 ] : NEW_LINE INDENT subset [ i ] [ j ] = subset [ i - 1 ] [ j ] NEW_LINE DEDENT if j >= set [ i - 1 ] : NEW_LINE INDENT subset [ i ] [ j ] = ( subset [ i - 1 ] [ j ] or subset [ i - 1 ] [ j - set [ i - 1 ] ] ) NEW_LINE DEDENT DEDENT DEDENT
for i in range ( n + 1 ) : NEW_LINE INDENT for j in range ( sum + 1 ) : NEW_LINE INDENT print ( subset [ i ] [ j ] , end = " ▁ " ) NEW_LINE print ( ) NEW_LINE DEDENT DEDENT return subset [ n ] [ sum ] NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT set = [ 3 , 34 , 4 , 12 , 5 , 2 ] NEW_LINE sum = 9 NEW_LINE n = len ( set ) NEW_LINE if ( isSubsetSum ( set , n , sum ) == True ) : NEW_LINE INDENT print ( " Found ▁ a ▁ subset ▁ with ▁ given ▁ sum " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No ▁ subset ▁ with ▁ given ▁ sum " ) NEW_LINE DEDENT DEDENT
def findoptimal ( N ) : NEW_LINE
if N <= 6 : NEW_LINE INDENT return N NEW_LINE DEDENT
maxi = 0 NEW_LINE
for b in range ( N - 3 , 0 , - 1 ) : NEW_LINE
curr = ( N - b - 1 ) * findoptimal ( b ) NEW_LINE if curr > maxi : NEW_LINE INDENT maxi = curr NEW_LINE DEDENT return maxi NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
for n in range ( 1 , 21 ) : NEW_LINE INDENT print ( ' Maximum ▁ Number ▁ of ▁ As ▁ with ▁ ' , n , ' keystrokes ▁ is ▁ ' , findoptimal ( n ) ) NEW_LINE DEDENT
def findoptimal ( N ) : NEW_LINE
if ( N <= 6 ) : NEW_LINE INDENT return N NEW_LINE DEDENT
screen = [ 0 ] * N NEW_LINE
for n in range ( 1 , 7 ) : NEW_LINE INDENT screen [ n - 1 ] = n NEW_LINE DEDENT
for n in range ( 7 , N + 1 ) : NEW_LINE
screen [ n - 1 ] = 0 NEW_LINE
for b in range ( n - 3 , 0 , - 1 ) : NEW_LINE
curr = ( n - b - 1 ) * screen [ b - 1 ] NEW_LINE if ( curr > screen [ n - 1 ] ) : NEW_LINE INDENT screen [ n - 1 ] = curr NEW_LINE DEDENT return screen [ N - 1 ] NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
for N in range ( 1 , 21 ) : NEW_LINE INDENT print ( " Maximum ▁ Number ▁ of ▁ A ' s ▁ with ▁ " , N , " ▁ keystrokes ▁ is ▁ " , findoptimal ( N ) ) NEW_LINE DEDENT
def findoptimal ( N ) : NEW_LINE
if ( N <= 6 ) : NEW_LINE INDENT return N NEW_LINE DEDENT
screen = [ 0 ] * N NEW_LINE
for n in range ( 1 , 7 ) : NEW_LINE INDENT screen [ n - 1 ] = n NEW_LINE DEDENT
for n in range ( 7 , N + 1 ) : NEW_LINE
screen [ n - 1 ] = max ( 2 * screen [ n - 4 ] , max ( 3 * screen [ n - 5 ] , 4 * screen [ n - 6 ] ) ) ; NEW_LINE return screen [ N - 1 ] NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
for N in range ( 1 , 21 ) : NEW_LINE INDENT print ( " Maximum ▁ Number ▁ of ▁ A ' s ▁ with ▁ " , N , " ▁ keystrokes ▁ is ▁ " , findoptimal ( N ) ) NEW_LINE DEDENT
def power ( x , y ) : NEW_LINE INDENT if ( y == 0 ) : return 1 NEW_LINE elif ( int ( y % 2 ) == 0 ) : NEW_LINE INDENT return ( power ( x , int ( y / 2 ) ) * power ( x , int ( y / 2 ) ) ) NEW_LINE DEDENT else : NEW_LINE INDENT return ( x * power ( x , int ( y / 2 ) ) * power ( x , int ( y / 2 ) ) ) NEW_LINE DEDENT DEDENT
x = 2 ; y = 3 NEW_LINE print ( power ( x , y ) ) NEW_LINE
def power ( x , y ) : NEW_LINE INDENT temp = 0 NEW_LINE if ( y == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT temp = power ( x , int ( y / 2 ) ) NEW_LINE if ( y % 2 == 0 ) : NEW_LINE INDENT return temp * temp ; NEW_LINE DEDENT else : NEW_LINE INDENT return x * temp * temp ; NEW_LINE DEDENT DEDENT
def power ( x , y ) : NEW_LINE INDENT if ( y == 0 ) : return 1 NEW_LINE temp = power ( x , int ( y / 2 ) ) NEW_LINE if ( y % 2 == 0 ) : NEW_LINE INDENT return temp * temp NEW_LINE DEDENT else : NEW_LINE INDENT if ( y > 0 ) : return x * temp * temp NEW_LINE else : return ( temp * temp ) / x NEW_LINE DEDENT DEDENT
x , y = 2 , - 3 NEW_LINE print ( ' % .6f ' % ( power ( x , y ) ) ) NEW_LINE
def power ( x , y ) : NEW_LINE
if ( y == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT
if ( x == 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
return x * power ( x , y - 1 ) NEW_LINE
x = 2 NEW_LINE y = 3 NEW_LINE print ( power ( x , y ) ) NEW_LINE
def squareRoot ( n ) : NEW_LINE
x = n NEW_LINE y = 1 NEW_LINE
e = 0.000001 NEW_LINE while ( x - y > e ) : NEW_LINE INDENT x = ( x + y ) / 2 NEW_LINE y = n / x NEW_LINE DEDENT return x NEW_LINE
n = 50 NEW_LINE print ( " Square ▁ root ▁ of " , n , " is " , round ( squareRoot ( n ) , 6 ) ) NEW_LINE
def getAvg ( prev_avg , x , n ) : NEW_LINE INDENT return ( ( prev_avg * n + x ) / ( n + 1 ) ) ; NEW_LINE DEDENT
def streamAvg ( arr , n ) : NEW_LINE INDENT avg = 0 ; NEW_LINE for i in range ( n ) : NEW_LINE INDENT avg = getAvg ( avg , arr [ i ] , i ) ; NEW_LINE print ( " Average ▁ of ▁ " , i + 1 , " ▁ numbers ▁ is ▁ " , avg ) ; NEW_LINE DEDENT DEDENT
arr = [ 10 , 20 , 30 , 40 , 50 , 60 ] ; NEW_LINE n = len ( arr ) ; NEW_LINE streamAvg ( arr , n ) ; NEW_LINE
def getAvg ( x , n , sum ) : NEW_LINE INDENT sum = sum + x ; NEW_LINE return float ( sum ) / n ; NEW_LINE DEDENT
def streamAvg ( arr , n ) : NEW_LINE INDENT avg = 0 ; NEW_LINE sum = 0 ; NEW_LINE for i in range ( n ) : NEW_LINE INDENT avg = getAvg ( arr [ i ] , i + 1 , sum ) ; NEW_LINE sum = avg * ( i + 1 ) ; NEW_LINE print ( " Average ▁ of ▁ " , end = " " ) ; NEW_LINE print ( i + 1 , end = " " ) ; NEW_LINE print ( " ▁ numbers ▁ is ▁ " , end = " " ) ; NEW_LINE print ( avg ) ; NEW_LINE DEDENT return ; NEW_LINE DEDENT
arr = [ 10 , 20 , 30 , 40 , 50 , 60 ] ; NEW_LINE n = len ( arr ) ; NEW_LINE streamAvg ( arr , n ) ; NEW_LINE
def binomialCoefficient ( n , k ) : NEW_LINE INDENT res = 1 NEW_LINE DEDENT
if ( k > n - k ) : NEW_LINE INDENT k = n - k NEW_LINE DEDENT
for i in range ( k ) : NEW_LINE INDENT res = res * ( n - i ) NEW_LINE res = res / ( i + 1 ) NEW_LINE DEDENT return res NEW_LINE
n = 8 NEW_LINE k = 2 NEW_LINE res = binomialCoefficient ( n , k ) NEW_LINE print ( " Value ▁ of ▁ C ( % ▁ d , ▁ % ▁ d ) ▁ is ▁ % ▁ d " % ( n , k , res ) ) NEW_LINE
import math NEW_LINE
def primeFactors ( n ) : NEW_LINE
while n % 2 == 0 : NEW_LINE INDENT print 2 , NEW_LINE n = n / 2 NEW_LINE DEDENT
for i in range ( 3 , int ( math . sqrt ( n ) ) + 1 , 2 ) : NEW_LINE
while n % i == 0 : NEW_LINE INDENT print i , NEW_LINE n = n / i NEW_LINE DEDENT
if n > 2 : NEW_LINE INDENT print n NEW_LINE DEDENT
n = 315 NEW_LINE primeFactors ( n ) NEW_LINE
def printCombination ( arr , n , r ) : NEW_LINE
data = [ 0 ] * r ; NEW_LINE
combinationUtil ( arr , data , 0 , n - 1 , 0 , r ) ; NEW_LINE
def combinationUtil ( arr , data , start , end , index , r ) : NEW_LINE
if ( index == r ) : NEW_LINE INDENT for j in range ( r ) : NEW_LINE INDENT print ( data [ j ] , end = " ▁ " ) ; NEW_LINE DEDENT print ( ) ; NEW_LINE return ; NEW_LINE DEDENT
i = start ; NEW_LINE while ( i <= end and end - i + 1 >= r - index ) : NEW_LINE INDENT data [ index ] = arr [ i ] ; NEW_LINE combinationUtil ( arr , data , i + 1 , end , index + 1 , r ) ; NEW_LINE i += 1 ; NEW_LINE DEDENT
arr = [ 1 , 2 , 3 , 4 , 5 ] ; NEW_LINE r = 3 ; NEW_LINE n = len ( arr ) ; NEW_LINE printCombination ( arr , n , r ) ; NEW_LINE
def printCombination ( arr , n , r ) : NEW_LINE
data = [ 0 ] * r NEW_LINE
combinationUtil ( arr , n , r , 0 , data , 0 ) NEW_LINE
def combinationUtil ( arr , n , r , index , data , i ) : NEW_LINE
if ( index == r ) : NEW_LINE INDENT for j in range ( r ) : NEW_LINE INDENT print ( data [ j ] , end = " ▁ " ) NEW_LINE DEDENT print ( ) NEW_LINE return NEW_LINE DEDENT
if ( i >= n ) : NEW_LINE INDENT return NEW_LINE DEDENT
data [ index ] = arr [ i ] NEW_LINE combinationUtil ( arr , n , r , index + 1 , data , i + 1 ) NEW_LINE
combinationUtil ( arr , n , r , index , data , i + 1 ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 1 , 2 , 3 , 4 , 5 ] NEW_LINE r = 3 NEW_LINE n = len ( arr ) NEW_LINE printCombination ( arr , n , r ) NEW_LINE DEDENT
def findgroups ( arr , n ) : NEW_LINE
c = [ 0 , 0 , 0 ] NEW_LINE
res = 0 NEW_LINE
for i in range ( 0 , n ) : NEW_LINE INDENT c [ arr [ i ] % 3 ] += 1 NEW_LINE DEDENT
res += ( ( c [ 0 ] * ( c [ 0 ] - 1 ) ) >> 1 ) NEW_LINE
res += c [ 1 ] * c [ 2 ] NEW_LINE
res += ( c [ 0 ] * ( c [ 0 ] - 1 ) * ( c [ 0 ] - 2 ) ) / 6 NEW_LINE
res += ( c [ 1 ] * ( c [ 1 ] - 1 ) * ( c [ 1 ] - 2 ) ) / 6 NEW_LINE
res += ( ( c [ 2 ] * ( c [ 2 ] - 1 ) * ( c [ 2 ] - 2 ) ) / 6 ) NEW_LINE
res += c [ 0 ] * c [ 1 ] * c [ 2 ] NEW_LINE
return res NEW_LINE
arr = [ 3 , 6 , 7 , 2 , 9 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Required ▁ number ▁ of ▁ groups ▁ are " , int ( findgroups ( arr , n ) ) ) NEW_LINE
def nextPowerOf2 ( n ) : NEW_LINE INDENT count = 0 ; NEW_LINE DEDENT
if ( n and not ( n & ( n - 1 ) ) ) : NEW_LINE INDENT return n NEW_LINE DEDENT while ( n != 0 ) : NEW_LINE INDENT n >>= 1 NEW_LINE count += 1 NEW_LINE DEDENT return 1 << count ; NEW_LINE
n = 0 NEW_LINE print ( nextPowerOf2 ( n ) ) NEW_LINE
def nextPowerOf2 ( n ) : NEW_LINE INDENT p = 1 NEW_LINE if ( n and not ( n & ( n - 1 ) ) ) : NEW_LINE INDENT return n NEW_LINE DEDENT while ( p < n ) : NEW_LINE INDENT p <<= 1 NEW_LINE DEDENT return p ; NEW_LINE DEDENT
n = 5 NEW_LINE print ( nextPowerOf2 ( n ) ) ; NEW_LINE
def nextPowerOf2 ( n ) : NEW_LINE INDENT n -= 1 NEW_LINE n |= n >> 1 NEW_LINE n |= n >> 2 NEW_LINE n |= n >> 4 NEW_LINE n |= n >> 8 NEW_LINE n |= n >> 16 NEW_LINE n += 1 NEW_LINE return n NEW_LINE DEDENT
n = 5 NEW_LINE print ( nextPowerOf2 ( n ) ) NEW_LINE
def segregate0and1 ( arr , n ) : NEW_LINE
count = 0 NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT if ( arr [ i ] == 0 ) : NEW_LINE INDENT count = count + 1 NEW_LINE DEDENT DEDENT
for i in range ( 0 , count ) : NEW_LINE INDENT arr [ i ] = 0 NEW_LINE DEDENT
for i in range ( count , n ) : NEW_LINE INDENT arr [ i ] = 1 NEW_LINE DEDENT
def print_arr ( arr , n ) : NEW_LINE INDENT print ( " Array ▁ after ▁ segregation ▁ is ▁ " , end = " " ) NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT DEDENT
arr = [ 0 , 1 , 0 , 1 , 1 , 1 ] NEW_LINE n = len ( arr ) NEW_LINE segregate0and1 ( arr , n ) NEW_LINE print_arr ( arr , n ) NEW_LINE
def segregate0and1 ( arr , size ) : NEW_LINE
left , right = 0 , size - 1 NEW_LINE while left < right : NEW_LINE
while arr [ left ] == 0 and left < right : NEW_LINE INDENT left += 1 NEW_LINE DEDENT
while arr [ right ] == 1 and left < right : NEW_LINE INDENT right -= 1 NEW_LINE DEDENT
if left < right : NEW_LINE INDENT arr [ left ] = 0 NEW_LINE arr [ right ] = 1 NEW_LINE left += 1 NEW_LINE right -= 1 NEW_LINE DEDENT return arr NEW_LINE
arr = [ 0 , 1 , 0 , 1 , 1 , 1 ] NEW_LINE arr_size = len ( arr ) NEW_LINE print ( " Array ▁ after ▁ segregation " ) NEW_LINE print ( segregate0and1 ( arr , arr_size ) ) NEW_LINE
def segregate0and1 ( arr , size ) : NEW_LINE INDENT type0 = 0 NEW_LINE type1 = size - 1 NEW_LINE while ( type0 < type1 ) : NEW_LINE INDENT if ( arr [ type0 ] == 1 ) : NEW_LINE INDENT ( arr [ type0 ] , arr [ type1 ] ) = ( arr [ type1 ] , arr [ type0 ] ) NEW_LINE type1 -= 1 NEW_LINE DEDENT else : NEW_LINE INDENT type0 += 1 NEW_LINE DEDENT DEDENT DEDENT
arr = [ 0 , 1 , 0 , 1 , 1 , 1 ] NEW_LINE arr_size = len ( arr ) NEW_LINE segregate0and1 ( arr , arr_size ) NEW_LINE print ( " Array ▁ after ▁ segregation ▁ is " , end = " ▁ " ) NEW_LINE for i in range ( 0 , arr_size ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT
def distantAdjacentElement ( a , n ) : NEW_LINE
m = dict ( ) NEW_LINE
for i in range ( n ) : NEW_LINE INDENT if a [ i ] in m : NEW_LINE INDENT m [ a [ i ] ] += 1 NEW_LINE DEDENT else : NEW_LINE INDENT m [ a [ i ] ] = 1 NEW_LINE DEDENT DEDENT
mx = 0 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT if mx < m [ a [ i ] ] : NEW_LINE INDENT mx = m [ a [ i ] ] NEW_LINE DEDENT DEDENT
if mx > ( n + 1 ) // 2 : NEW_LINE INDENT print ( " NO " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " YES " ) NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = [ 7 , 7 , 7 , 7 ] NEW_LINE n = len ( a ) NEW_LINE distantAdjacentElement ( a , n ) NEW_LINE DEDENT
def maxIndexDiff ( arr , n ) : NEW_LINE INDENT maxDiff = - 1 NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT j = n - 1 NEW_LINE while ( j > i ) : NEW_LINE INDENT if arr [ j ] > arr [ i ] and maxDiff < ( j - i ) : NEW_LINE INDENT maxDiff = j - i NEW_LINE DEDENT j -= 1 NEW_LINE DEDENT DEDENT return maxDiff NEW_LINE DEDENT
arr = [ 9 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 18 , 0 ] NEW_LINE n = len ( arr ) NEW_LINE maxDiff = maxIndexDiff ( arr , n ) NEW_LINE print ( maxDiff ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT v = [ 34 , 8 , 10 , 3 , 2 , 80 , 30 , 33 , 1 ] ; NEW_LINE n = len ( v ) ; NEW_LINE maxFromEnd = [ - 38749432 ] * ( n + 1 ) ; NEW_LINE DEDENT
for i in range ( n - 1 , 0 , - 1 ) : NEW_LINE INDENT maxFromEnd [ i ] = max ( maxFromEnd [ i + 1 ] , v [ i ] ) ; NEW_LINE DEDENT result = 0 ; NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT low = i + 1 ; high = n - 1 ; ans = i ; NEW_LINE while ( low <= high ) : NEW_LINE INDENT mid = int ( ( low + high ) / 2 ) ; NEW_LINE if ( v [ i ] <= maxFromEnd [ mid ] ) : NEW_LINE DEDENT DEDENT
ans = max ( ans , mid ) ; NEW_LINE low = mid + 1 ; NEW_LINE else : NEW_LINE high = mid - 1 ; NEW_LINE
result = max ( result , ans - i ) ; NEW_LINE print ( result , end = " " ) ; NEW_LINE
def printRepeating ( arr , size ) : NEW_LINE
s = set ( ) NEW_LINE for i in range ( size ) : NEW_LINE INDENT if arr [ i ] not in s : NEW_LINE INDENT s . add ( arr [ i ] ) NEW_LINE DEDENT DEDENT
for i in s : NEW_LINE INDENT print ( i , end = " ▁ " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 1 , 3 , 2 , 2 , 1 ] NEW_LINE size = len ( arr ) NEW_LINE printRepeating ( arr , size ) NEW_LINE DEDENT
def minSwapsToSort ( arr , n ) : NEW_LINE
arrPos = [ [ 0 for x in range ( 2 ) ] for y in range ( n ) ] NEW_LINE for i in range ( n ) : NEW_LINE INDENT arrPos [ i ] [ 0 ] = arr [ i ] NEW_LINE arrPos [ i ] [ 1 ] = i NEW_LINE DEDENT
arrPos . sort ( ) NEW_LINE
vis = [ False ] * ( n ) NEW_LINE
ans = 0 NEW_LINE
for i in range ( n ) : NEW_LINE
if ( vis [ i ] or arrPos [ i ] [ 1 ] == i ) : NEW_LINE INDENT continue NEW_LINE DEDENT
cycle_size = 0 NEW_LINE j = i NEW_LINE while ( not vis [ j ] ) : NEW_LINE INDENT vis [ j ] = 1 NEW_LINE DEDENT
j = arrPos [ j ] [ 1 ] NEW_LINE cycle_size += 1 NEW_LINE
ans += ( cycle_size - 1 ) NEW_LINE
return ans NEW_LINE
def minSwapToMakeArraySame ( a , b , n ) : NEW_LINE
mp = { } NEW_LINE for i in range ( n ) : NEW_LINE INDENT mp [ b [ i ] ] = i NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT b [ i ] = mp [ a [ i ] ] NEW_LINE DEDENT
return minSwapsToSort ( b , n ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = [ 3 , 6 , 4 , 8 ] NEW_LINE b = [ 4 , 6 , 8 , 3 ] NEW_LINE n = len ( a ) NEW_LINE print ( minSwapToMakeArraySame ( a , b , n ) ) NEW_LINE DEDENT
def missingK ( a , k , n ) : NEW_LINE INDENT difference = 0 NEW_LINE ans = 0 NEW_LINE count = k NEW_LINE flag = 0 NEW_LINE DEDENT
for i in range ( 0 , n - 1 ) : NEW_LINE INDENT difference = 0 NEW_LINE DEDENT
if ( ( a [ i ] + 1 ) != a [ i + 1 ] ) : NEW_LINE
difference += ( a [ i + 1 ] - a [ i ] ) - 1 NEW_LINE
if ( difference >= count ) : NEW_LINE INDENT ans = a [ i ] + count NEW_LINE flag = 1 NEW_LINE break NEW_LINE DEDENT else : NEW_LINE INDENT count -= difference NEW_LINE DEDENT
if ( flag ) : NEW_LINE INDENT return ans NEW_LINE DEDENT else : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
a = [ 1 , 5 , 11 , 19 ] NEW_LINE
k = 11 NEW_LINE n = len ( a ) NEW_LINE
missing = missingK ( a , k , n ) NEW_LINE print ( missing ) NEW_LINE
def missingK ( arr , k ) : NEW_LINE INDENT n = len ( arr ) NEW_LINE l = 0 NEW_LINE u = n - 1 NEW_LINE mid = 0 NEW_LINE while ( l <= u ) : NEW_LINE INDENT mid = ( l + u ) // 2 ; NEW_LINE numbers_less_than_mid = arr [ mid ] - ( mid + 1 ) ; NEW_LINE DEDENT DEDENT
if ( numbers_less_than_mid == k ) : NEW_LINE
if ( mid > 0 and ( arr [ mid - 1 ] - ( mid ) ) == k ) : NEW_LINE INDENT u = mid - 1 ; NEW_LINE continue ; NEW_LINE DEDENT
return arr [ mid ] - 1 ; NEW_LINE
if ( numbers_less_than_mid < k ) : NEW_LINE l = mid + 1 ; NEW_LINE elif ( k < numbers_less_than_mid ) : NEW_LINE u = mid - 1 ; NEW_LINE
INDENT if ( u < 0 ) : NEW_LINE INDENT return k ; NEW_LINE DEDENT DEDENT
INDENT less = arr [ u ] - ( u + 1 ) ; NEW_LINE k -= less ; NEW_LINE DEDENT
INDENT return arr [ u ] + k ; NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 2 , 3 , 4 , 7 , 11 ] ; NEW_LINE k = 5 ; NEW_LINE DEDENT
print ( " Missing ▁ kth ▁ number ▁ = ▁ " + str ( missingK ( arr , k ) ) ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . next = next NEW_LINE DEDENT DEDENT
def printList ( node ) : NEW_LINE INDENT while ( node != None ) : NEW_LINE INDENT print ( node . data , end = " ▁ " ) NEW_LINE node = node . next NEW_LINE DEDENT print ( " " ) NEW_LINE DEDENT
def newNode ( key ) : NEW_LINE INDENT temp = Node ( 0 ) NEW_LINE temp . data = key NEW_LINE temp . next = None NEW_LINE return temp NEW_LINE DEDENT
def insertBeg ( head , val ) : NEW_LINE INDENT temp = newNode ( val ) NEW_LINE temp . next = head NEW_LINE head = temp NEW_LINE return head NEW_LINE DEDENT
def rearrangeOddEven ( head ) : NEW_LINE INDENT odd = [ ] NEW_LINE even = [ ] NEW_LINE i = 1 NEW_LINE while ( head != None ) : NEW_LINE INDENT if ( head . data % 2 != 0 and i % 2 == 0 ) : NEW_LINE DEDENT DEDENT
odd . append ( head ) NEW_LINE elif ( head . data % 2 == 0 and i % 2 != 0 ) : NEW_LINE
even . append ( head ) NEW_LINE head = head . next NEW_LINE i = i + 1 NEW_LINE while ( len ( odd ) != 0 and len ( even ) != 0 ) : NEW_LINE
odd [ - 1 ] . data , even [ - 1 ] . data = even [ - 1 ] . data , odd [ - 1 ] . data NEW_LINE odd . pop ( ) NEW_LINE even . pop ( ) NEW_LINE return head NEW_LINE
head = newNode ( 8 ) NEW_LINE head = insertBeg ( head , 7 ) NEW_LINE head = insertBeg ( head , 6 ) NEW_LINE head = insertBeg ( head , 5 ) NEW_LINE head = insertBeg ( head , 3 ) NEW_LINE head = insertBeg ( head , 2 ) NEW_LINE head = insertBeg ( head , 1 ) NEW_LINE print ( " Linked ▁ List : " ) NEW_LINE printList ( head ) NEW_LINE rearrangeOddEven ( head ) NEW_LINE print ( " Linked ▁ List ▁ after ▁ " , " Rearranging : " ) NEW_LINE printList ( head ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self ) : NEW_LINE INDENT self . data = 0 NEW_LINE self . next = None NEW_LINE DEDENT DEDENT
def printList ( node ) : NEW_LINE INDENT while ( node != None ) : NEW_LINE INDENT print ( node . data , end = " ▁ " ) NEW_LINE node = node . next NEW_LINE DEDENT print ( " ▁ " ) NEW_LINE DEDENT
def newNode ( key ) : NEW_LINE INDENT temp = Node ( ) NEW_LINE temp . data = key NEW_LINE temp . next = None NEW_LINE return temp NEW_LINE DEDENT
def insertBeg ( head , val ) : NEW_LINE INDENT temp = newNode ( val ) NEW_LINE temp . next = head NEW_LINE head = temp NEW_LINE return head NEW_LINE DEDENT
def rearrange ( head ) : NEW_LINE
even = None NEW_LINE temp = None NEW_LINE prev_temp = None NEW_LINE i = None NEW_LINE j = None NEW_LINE k = None NEW_LINE l = None NEW_LINE ptr = None NEW_LINE
temp = ( head ) . next NEW_LINE prev_temp = head NEW_LINE while ( temp != None ) : NEW_LINE
x = temp . next NEW_LINE
if ( temp . data % 2 != 0 ) : NEW_LINE INDENT prev_temp . next = x NEW_LINE temp . next = ( head ) NEW_LINE ( head ) = temp NEW_LINE DEDENT else : NEW_LINE INDENT prev_temp = temp NEW_LINE DEDENT
temp = x NEW_LINE
temp = ( head ) . next NEW_LINE prev_temp = ( head ) NEW_LINE while ( temp != None and temp . data % 2 != 0 ) : NEW_LINE INDENT prev_temp = temp NEW_LINE temp = temp . next NEW_LINE DEDENT even = temp NEW_LINE
prev_temp . next = None NEW_LINE
i = head NEW_LINE j = even NEW_LINE while ( j != None and i != None ) : NEW_LINE
k = i . next NEW_LINE l = j . next NEW_LINE i . next = j NEW_LINE j . next = k NEW_LINE
ptr = j NEW_LINE
i = k NEW_LINE j = l NEW_LINE if ( i == None ) : NEW_LINE
ptr . next = j NEW_LINE
return head NEW_LINE
head = newNode ( 8 ) NEW_LINE head = insertBeg ( head , 7 ) NEW_LINE head = insertBeg ( head , 6 ) NEW_LINE head = insertBeg ( head , 3 ) NEW_LINE head = insertBeg ( head , 5 ) NEW_LINE head = insertBeg ( head , 1 ) NEW_LINE head = insertBeg ( head , 2 ) NEW_LINE head = insertBeg ( head , 10 ) NEW_LINE print ( " Linked ▁ List : " ) NEW_LINE printList ( head ) NEW_LINE print ( " Rearranged ▁ List " ) NEW_LINE head = rearrange ( head ) NEW_LINE printList ( head ) NEW_LINE
def printMat ( mat ) : NEW_LINE
for i in range ( len ( mat ) ) : NEW_LINE
for j in range ( len ( mat [ 0 ] ) ) : NEW_LINE
print ( mat [ i ] [ j ] , end = " ▁ " ) NEW_LINE print ( ) NEW_LINE
def performSwap ( mat , i , j ) : NEW_LINE INDENT N = len ( mat ) NEW_LINE DEDENT
ei = N - 1 - i NEW_LINE
ej = N - 1 - j NEW_LINE
temp = mat [ i ] [ j ] NEW_LINE mat [ i ] [ j ] = mat [ ej ] [ i ] NEW_LINE mat [ ej ] [ i ] = mat [ ei ] [ ej ] NEW_LINE mat [ ei ] [ ej ] = mat [ j ] [ ei ] NEW_LINE mat [ j ] [ ei ] = temp NEW_LINE
def rotate ( mat , N , K ) : NEW_LINE
K = K % 4 NEW_LINE
while ( K > 0 ) : NEW_LINE
for i in range ( int ( N / 2 ) ) : NEW_LINE
for j in range ( i , N - i - 1 ) : NEW_LINE
if ( i != j and ( i + j ) != N - 1 ) : NEW_LINE
performSwap ( mat , i , j ) NEW_LINE K -= 1 NEW_LINE
printMat ( mat ) NEW_LINE
K = 5 NEW_LINE mat = [ [ 1 , 2 , 3 , 4 ] , [ 6 , 7 , 8 , 9 ] , [ 11 , 12 , 13 , 14 ] , [ 16 , 17 , 18 , 19 ] ] NEW_LINE N = len ( mat ) NEW_LINE rotate ( mat , N , K ) NEW_LINE
def findRotations ( str ) : NEW_LINE
tmp = str + str NEW_LINE n = len ( str ) NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE
substring = tmp [ i : i + n ] NEW_LINE
if ( str == substring ) : NEW_LINE INDENT return i NEW_LINE DEDENT return n NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT str = " abc " NEW_LINE print ( findRotations ( str ) ) NEW_LINE DEDENT
MAX = 10000 NEW_LINE
prefix = [ 0 ] * ( MAX + 1 ) NEW_LINE def isPowerOfTwo ( x ) : NEW_LINE INDENT if ( x and ( not ( x & ( x - 1 ) ) ) ) : NEW_LINE INDENT return True NEW_LINE DEDENT return False NEW_LINE DEDENT
def computePrefix ( n , a ) : NEW_LINE
if ( isPowerOfTwo ( a [ 0 ] ) ) : NEW_LINE INDENT prefix [ 0 ] = 1 NEW_LINE DEDENT for i in range ( 1 , n ) : NEW_LINE INDENT prefix [ i ] = prefix [ i - 1 ] NEW_LINE if ( isPowerOfTwo ( a [ i ] ) ) : NEW_LINE INDENT prefix [ i ] += 1 NEW_LINE DEDENT DEDENT
def query ( L , R ) : NEW_LINE INDENT return prefix [ R ] - prefix [ L - 1 ] NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT A = [ 3 , 8 , 5 , 2 , 5 , 10 ] NEW_LINE N = len ( A ) NEW_LINE Q = 2 NEW_LINE computePrefix ( N , A ) NEW_LINE print ( query ( 0 , 4 ) ) NEW_LINE print ( query ( 3 , 5 ) ) NEW_LINE DEDENT
def countIntgralPoints ( x1 , y1 , x2 , y2 ) : NEW_LINE INDENT print ( ( y2 - y1 - 1 ) * ( x2 - x1 - 1 ) ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT x1 = 1 NEW_LINE y1 = 1 NEW_LINE x2 = 4 NEW_LINE y2 = 4 NEW_LINE countIntgralPoints ( x1 , y1 , x2 , y2 ) NEW_LINE DEDENT
def findNextNumber ( n ) : NEW_LINE INDENT h = [ 0 for i in range ( 10 ) ] NEW_LINE i = 0 NEW_LINE msb = n NEW_LINE rem = 0 NEW_LINE next_num = - 1 NEW_LINE count = 0 NEW_LINE DEDENT
while ( msb > 9 ) : NEW_LINE INDENT rem = msb % 10 NEW_LINE h [ rem ] = 1 NEW_LINE msb //= 10 NEW_LINE count += 1 NEW_LINE DEDENT h [ msb ] = 1 NEW_LINE count += 1 NEW_LINE
for i in range ( msb + 1 , 10 , 1 ) : NEW_LINE INDENT if ( h [ i ] == 0 ) : NEW_LINE INDENT next_num = i NEW_LINE break NEW_LINE DEDENT DEDENT
if ( next_num == - 1 ) : NEW_LINE INDENT for i in range ( 1 , msb , 1 ) : NEW_LINE INDENT if ( h [ i ] == 0 ) : NEW_LINE INDENT next_num = i NEW_LINE count += 1 NEW_LINE break NEW_LINE DEDENT DEDENT DEDENT
if ( next_num > 0 ) : NEW_LINE
for i in range ( 0 , 10 , 1 ) : NEW_LINE INDENT if ( h [ i ] == 0 ) : NEW_LINE INDENT msb = i NEW_LINE break NEW_LINE DEDENT DEDENT
for i in range ( 1 , count , 1 ) : NEW_LINE INDENT next_num = ( ( next_num * 10 ) + msb ) NEW_LINE DEDENT
if ( next_num > n ) : NEW_LINE INDENT print ( next_num ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Not ▁ Possible " ) NEW_LINE DEDENT else : NEW_LINE print ( " Not ▁ Possible " ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 2019 NEW_LINE findNextNumber ( n ) NEW_LINE DEDENT
def CalculateValues ( N ) : NEW_LINE
for C in range ( 0 , N // 7 + 1 ) : NEW_LINE
for B in range ( 0 , N // 5 + 1 ) : NEW_LINE
A = N - 7 * C - 5 * B NEW_LINE
if ( A >= 0 and A % 3 == 0 ) : NEW_LINE INDENT print ( " A ▁ = " , A / 3 , " , ▁ B ▁ = " , B , " , ▁ \ ▁ C ▁ = " , C , sep = " ▁ " ) NEW_LINE return NEW_LINE DEDENT
print ( - 1 ) NEW_LINE return NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 19 NEW_LINE CalculateValues ( 19 ) NEW_LINE DEDENT
def minimumTime ( arr , n ) : NEW_LINE
sum = 0 NEW_LINE
T = max ( arr ) NEW_LINE
for i in range ( n ) : NEW_LINE
sum += arr [ i ] NEW_LINE
print ( max ( 2 * T , sum ) ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 2 , 8 , 3 ] NEW_LINE N = len ( arr ) NEW_LINE DEDENT
minimumTime ( arr , N ) NEW_LINE
def lexicographicallyMax ( s ) : NEW_LINE
n = len ( s ) NEW_LINE
for i in range ( n ) : NEW_LINE
count = 0 NEW_LINE
beg = i NEW_LINE
end = i NEW_LINE
if ( s [ i ] == '1' ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT
for j in range ( i + 1 , n ) : NEW_LINE INDENT if ( s [ j ] == '1' ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT if ( count % 2 == 0 and count != 0 ) : NEW_LINE INDENT end = j NEW_LINE break NEW_LINE DEDENT DEDENT
temp = s [ beg : end + 1 ] NEW_LINE temp = temp [ : : - 1 ] NEW_LINE s = s [ 0 : beg ] + temp + s [ end + 1 : ] NEW_LINE
print ( s ) NEW_LINE
S = "0101" NEW_LINE lexicographicallyMax ( S ) NEW_LINE
def maxPairs ( nums , k ) : NEW_LINE
nums = sorted ( nums ) NEW_LINE
result = 0 NEW_LINE
start , end = 0 , len ( nums ) - 1 NEW_LINE
while ( start < end ) : NEW_LINE INDENT if ( nums [ start ] + nums [ end ] > k ) : NEW_LINE DEDENT
end -= 1 NEW_LINE elif ( nums [ start ] + nums [ end ] < k ) : NEW_LINE
start += 1 NEW_LINE
else : NEW_LINE INDENT start += 1 NEW_LINE end -= 1 NEW_LINE result += 1 NEW_LINE DEDENT
print ( result ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 1 , 2 , 3 , 4 ] NEW_LINE K = 5 NEW_LINE DEDENT
maxPairs ( arr , K ) NEW_LINE
def maxPairs ( nums , k ) : NEW_LINE
m = { } NEW_LINE
result = 0 NEW_LINE
for i in nums : NEW_LINE
if ( ( i in m ) and m [ i ] > 0 ) : NEW_LINE INDENT m [ i ] = m [ i ] - 1 NEW_LINE result += 1 NEW_LINE DEDENT
else : NEW_LINE INDENT if k - i in m : NEW_LINE INDENT m [ k - i ] += 1 NEW_LINE DEDENT else : NEW_LINE INDENT m [ k - i ] = 1 NEW_LINE DEDENT DEDENT
print ( result ) NEW_LINE
arr = [ 1 , 2 , 3 , 4 ] NEW_LINE K = 5 NEW_LINE
maxPairs ( arr , K ) NEW_LINE
def removeIndicesToMakeSumEqual ( arr ) : NEW_LINE
N = len ( arr ) ; NEW_LINE
odd = [ 0 ] * N ; NEW_LINE
even = [ 0 ] * N ; NEW_LINE
even [ 0 ] = arr [ 0 ] ; NEW_LINE
for i in range ( 1 , N ) : NEW_LINE
odd [ i ] = odd [ i - 1 ] ; NEW_LINE
even [ i ] = even [ i - 1 ] ; NEW_LINE
if ( i % 2 == 0 ) : NEW_LINE
even [ i ] += arr [ i ] ; NEW_LINE
else : NEW_LINE
odd [ i ] += arr [ i ] ; NEW_LINE
find = False ; NEW_LINE
p = odd [ N - 1 ] ; NEW_LINE
q = even [ N - 1 ] - arr [ 0 ] ; NEW_LINE
if ( p == q ) : NEW_LINE INDENT print ( "0 ▁ " ) ; NEW_LINE find = True ; NEW_LINE DEDENT
for i in range ( 1 , N ) : NEW_LINE
if ( i % 2 == 0 ) : NEW_LINE
p = even [ N - 1 ] - even [ i - 1 ] - arr [ i ] + odd [ i - 1 ] ; NEW_LINE
q = odd [ N - 1 ] - odd [ i - 1 ] + even [ i - 1 ] ; NEW_LINE else : NEW_LINE
q = odd [ N - 1 ] - odd [ i - 1 ] - arr [ i ] + even [ i - 1 ] ; NEW_LINE
p = even [ N - 1 ] - even [ i - 1 ] + odd [ i - 1 ] ; NEW_LINE
if ( p == q ) : NEW_LINE
find = True ; NEW_LINE
print ( i , end = " " ) ; NEW_LINE
if ( find == False ) : NEW_LINE
print ( - 1 ) ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 4 , 1 , 6 , 2 ] ; NEW_LINE removeIndicesToMakeSumEqual ( arr ) ; NEW_LINE DEDENT
def min_element_removal ( arr , N ) : NEW_LINE
left = [ 1 ] * N NEW_LINE
right = [ 1 ] * ( N ) NEW_LINE
for i in range ( 1 , N ) : NEW_LINE
for j in range ( i ) : NEW_LINE
if ( arr [ j ] < arr [ i ] ) : NEW_LINE
left [ i ] = max ( left [ i ] , left [ j ] + 1 ) NEW_LINE
for i in range ( N - 2 , - 1 , - 1 ) : NEW_LINE
for j in range ( N - 1 , i , - 1 ) : NEW_LINE
if ( arr [ i ] > arr [ j ] ) : NEW_LINE
right [ i ] = max ( right [ i ] , right [ j ] + 1 ) NEW_LINE
maxLen = 0 NEW_LINE
for i in range ( 1 , N - 1 ) : NEW_LINE
maxLen = max ( maxLen , left [ i ] + right [ i ] - 1 ) NEW_LINE print ( ( N - maxLen ) ) NEW_LINE
def makeBitonic ( arr , N ) : NEW_LINE INDENT if ( N == 1 ) : NEW_LINE INDENT print ( "0" ) NEW_LINE return NEW_LINE DEDENT if ( N == 2 ) : NEW_LINE INDENT if ( arr [ 0 ] != arr [ 1 ] ) : NEW_LINE INDENT print ( "0" ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( "1" ) NEW_LINE DEDENT return NEW_LINE DEDENT min_element_removal ( arr , N ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 2 , 1 , 1 , 5 , 6 , 2 , 3 , 1 ] NEW_LINE N = len ( arr ) NEW_LINE makeBitonic ( arr , N ) NEW_LINE DEDENT
def countSubarrays ( A , N ) : NEW_LINE
ans = 0 ; NEW_LINE for i in range ( N - 1 ) : NEW_LINE
if ( A [ i ] != A [ i + 1 ] ) : NEW_LINE
ans += 1 ; NEW_LINE
j = i - 1 ; k = i + 2 ; NEW_LINE while ( j >= 0 and k < N and A [ j ] == A [ i ] and A [ k ] == A [ i + 1 ] ) : NEW_LINE
ans += 1 ; NEW_LINE j -= 1 ; NEW_LINE k += 1 ; NEW_LINE
print ( ans ) ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT A = [ 1 , 1 , 0 , 0 , 1 , 0 ] ; NEW_LINE N = len ( A ) ; NEW_LINE DEDENT
countSubarrays ( A , N ) ; NEW_LINE
maxN = 2002 NEW_LINE
lcount = [ [ 0 for i in range ( maxN ) ] for j in range ( maxN ) ] NEW_LINE
rcount = [ [ 0 for i in range ( maxN ) ] for j in range ( maxN ) ] NEW_LINE
def fill_counts ( a , n ) : NEW_LINE
maxA = a [ 0 ] NEW_LINE for i in range ( n ) : NEW_LINE INDENT if ( a [ i ] > maxA ) : NEW_LINE INDENT maxA = a [ i ] NEW_LINE DEDENT DEDENT for i in range ( n ) : NEW_LINE INDENT lcount [ a [ i ] ] [ i ] = 1 NEW_LINE rcount [ a [ i ] ] [ i ] = 1 NEW_LINE DEDENT for i in range ( maxA + 1 ) : NEW_LINE
for j in range ( n ) : NEW_LINE INDENT lcount [ i ] [ j ] = ( lcount [ i ] [ j - 1 ] + lcount [ i ] [ j ] ) NEW_LINE DEDENT
for j in range ( n - 2 , - 1 , - 1 ) : NEW_LINE INDENT rcount [ i ] [ j ] = ( rcount [ i ] [ j + 1 ] + rcount [ i ] [ j ] ) NEW_LINE DEDENT
def countSubsequence ( a , n ) : NEW_LINE INDENT fill_counts ( a , n ) NEW_LINE answer = 0 NEW_LINE for i in range ( 1 , n ) : NEW_LINE INDENT for j in range ( i + 1 , n - 1 ) : NEW_LINE INDENT answer += ( lcount [ a [ j ] ] [ i - 1 ] * rcount [ a [ i ] ] [ j + 1 ] ) NEW_LINE DEDENT DEDENT return answer NEW_LINE DEDENT
a = [ 1 , 2 , 3 , 2 , 1 , 3 , 2 ] NEW_LINE print ( countSubsequence ( a , 7 ) ) NEW_LINE
def removeOuterParentheses ( S ) : NEW_LINE
res = " " NEW_LINE
count = 0 NEW_LINE
for c in S : NEW_LINE
if ( c == ' ( ' and count > 0 ) : NEW_LINE
res += c NEW_LINE
if ( c == ' ( ' ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT if ( c == ' ) ' and count > 1 ) : NEW_LINE
res += c NEW_LINE if ( c == ' ) ' ) : NEW_LINE count -= 1 NEW_LINE
return res NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT S = " ( ( ) ( ) ) ( ( ) ) ( ) " NEW_LINE print ( removeOuterParentheses ( S ) ) NEW_LINE DEDENT
def maxiConsecutiveSubarray ( arr , N ) : NEW_LINE
maxi = 0 ; NEW_LINE for i in range ( N - 1 ) : NEW_LINE
cnt = 1 ; NEW_LINE for j in range ( i , N - 1 ) : NEW_LINE
if ( arr [ j + 1 ] == arr [ j ] + 1 ) : NEW_LINE INDENT cnt += 1 ; NEW_LINE DEDENT
else : NEW_LINE INDENT break ; NEW_LINE DEDENT
maxi = max ( maxi , cnt ) ; NEW_LINE i = j ; NEW_LINE
return maxi ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 11 ; NEW_LINE arr = [ 1 , 3 , 4 , 2 , 3 , 4 , 2 , 3 , 5 , 6 , 7 ] ; NEW_LINE print ( maxiConsecutiveSubarray ( arr , N ) ) ; NEW_LINE DEDENT
N = 100005 NEW_LINE
def SieveOfEratosthenes ( prime , p_size ) : NEW_LINE
prime [ 0 ] = False NEW_LINE prime [ 1 ] = False NEW_LINE p = 2 NEW_LINE while p * p <= p_size : NEW_LINE
if ( prime [ p ] ) : NEW_LINE
for i in range ( p * 2 , p_size + 1 , p ) : NEW_LINE INDENT prime [ i ] = False NEW_LINE DEDENT p += 1 NEW_LINE
def digitSum ( number ) : NEW_LINE
sum = 0 NEW_LINE while ( number > 0 ) : NEW_LINE
sum += ( number % 10 ) NEW_LINE number //= 10 NEW_LINE
return sum NEW_LINE
def longestCompositeDigitSumSubsequence ( arr , n ) : NEW_LINE INDENT count = 0 NEW_LINE prime = [ True ] * ( N + 1 ) NEW_LINE SieveOfEratosthenes ( prime , N ) NEW_LINE for i in range ( n ) : NEW_LINE DEDENT
res = digitSum ( arr [ i ] ) NEW_LINE
if ( res == 1 ) : NEW_LINE INDENT continue NEW_LINE DEDENT
if ( not prime [ res ] ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT print ( count ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 13 , 55 , 7 , 3 , 5 , 1 , 10 , 21 , 233 , 144 , 89 ] NEW_LINE n = len ( arr ) NEW_LINE DEDENT
longestCompositeDigitSumSubsequence ( arr , n ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT
def newnode ( data ) : NEW_LINE INDENT temp = Node ( data ) NEW_LINE DEDENT
return temp NEW_LINE
def insert ( s , i , N , root , temp ) : NEW_LINE INDENT if ( i == N ) : NEW_LINE INDENT return temp NEW_LINE DEDENT DEDENT
if ( s [ i ] == ' L ' ) : NEW_LINE INDENT root . left = insert ( s , i + 1 , N , root . left , temp ) NEW_LINE DEDENT
else : NEW_LINE INDENT root . right = insert ( s , i + 1 , N , root . right , temp ) NEW_LINE DEDENT
return root NEW_LINE
def SBTUtil ( root , sum ) : NEW_LINE
if ( root == None ) : NEW_LINE INDENT return [ 0 , sum ] NEW_LINE DEDENT if ( root . left == None and root . right == None ) : NEW_LINE INDENT return [ root . data , sum ] NEW_LINE DEDENT
left , sum = SBTUtil ( root . left , sum ) NEW_LINE
right , sum = SBTUtil ( root . right , sum ) NEW_LINE
if ( root . left and root . right ) : NEW_LINE
if ( ( left % 2 == 0 and right % 2 != 0 ) or ( left % 2 != 0 and right % 2 == 0 ) ) : NEW_LINE INDENT sum += root . data NEW_LINE DEDENT
return [ left + right + root . data , sum ] NEW_LINE
def build_tree ( R , N , str , values ) : NEW_LINE
root = newnode ( R ) NEW_LINE
for i in range ( 0 , N - 1 ) : NEW_LINE INDENT s = str [ i ] NEW_LINE x = values [ i ] NEW_LINE DEDENT
temp = newnode ( x ) NEW_LINE
root = insert ( s , 0 , len ( s ) , root , temp ) NEW_LINE
return root NEW_LINE
def speciallyBalancedNodes ( R , N , str , values ) : NEW_LINE
root = build_tree ( R , N , str , values ) NEW_LINE
sum = 0 NEW_LINE
tmp , sum = SBTUtil ( root , sum ) NEW_LINE
print ( sum , end = ' ▁ ' ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
N = 7 NEW_LINE
R = 12 NEW_LINE
str = [ " L " , " R " , " RL " , " RR " , " RLL " , " RLR " ] NEW_LINE
values = [ 17 , 16 , 4 , 9 , 2 , 3 ] NEW_LINE
speciallyBalancedNodes ( R , N , str , values ) NEW_LINE
def position ( arr , N ) : NEW_LINE
pos = - 1 ; NEW_LINE
count = 0 ; NEW_LINE
for i in range ( N ) : NEW_LINE
count = 0 ; NEW_LINE for j in range ( N ) : NEW_LINE
if ( arr [ i ] [ 0 ] <= arr [ j ] [ 0 ] and arr [ i ] [ 1 ] >= arr [ j ] [ 1 ] ) : NEW_LINE INDENT count += 1 ; NEW_LINE DEDENT
if ( count == N ) : NEW_LINE INDENT pos = i ; NEW_LINE DEDENT
if ( pos == - 1 ) : NEW_LINE INDENT print ( pos ) ; NEW_LINE DEDENT
else : NEW_LINE INDENT print ( pos + 1 ) ; NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE
arr = [ [ 3 , 3 ] , [ 1 , 3 ] , [ 2 , 2 ] , [ 2 , 3 ] , [ 1 , 2 ] ] ; NEW_LINE N = len ( arr ) ; NEW_LINE
position ( arr , N ) ; NEW_LINE
import sys NEW_LINE
def position ( arr , N ) : NEW_LINE
pos = - 1 NEW_LINE
right = - sys . maxsize - 1 NEW_LINE
left = sys . maxsize NEW_LINE
for i in range ( N ) : NEW_LINE
if ( arr [ i ] [ 1 ] > right ) : NEW_LINE INDENT right = arr [ i ] [ 1 ] NEW_LINE DEDENT
if ( arr [ i ] [ 0 ] < left ) : NEW_LINE INDENT left = arr [ i ] [ 0 ] NEW_LINE DEDENT
for i in range ( N ) : NEW_LINE
if ( arr [ i ] [ 0 ] == left and arr [ i ] [ 1 ] == right ) : NEW_LINE INDENT pos = i + 1 NEW_LINE DEDENT
print ( pos ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
arr = [ [ 3 , 3 ] , [ 1 , 3 ] , [ 2 , 2 ] , [ 2 , 3 ] , [ 1 , 2 ] ] NEW_LINE N = len ( arr ) NEW_LINE
position ( arr , N ) NEW_LINE
def ctMinEdits ( str1 , str2 ) : NEW_LINE INDENT N1 = len ( str1 ) NEW_LINE N2 = len ( str2 ) NEW_LINE DEDENT
freq1 = [ 0 ] * 256 NEW_LINE for i in range ( N1 ) : NEW_LINE INDENT freq1 [ ord ( str1 [ i ] ) ] += 1 NEW_LINE DEDENT
freq2 = [ 0 ] * 256 NEW_LINE for i in range ( N2 ) : NEW_LINE INDENT freq2 [ ord ( str2 [ i ] ) ] += 1 NEW_LINE DEDENT
for i in range ( 256 ) : NEW_LINE
if ( freq1 [ i ] > freq2 [ i ] ) : NEW_LINE INDENT freq1 [ i ] = freq1 [ i ] - freq2 [ i ] NEW_LINE freq2 [ i ] = 0 NEW_LINE DEDENT
else : NEW_LINE INDENT freq2 [ i ] = freq2 [ i ] - freq1 [ i ] NEW_LINE freq1 [ i ] = 0 NEW_LINE DEDENT
sum1 = 0 NEW_LINE
sum2 = 0 NEW_LINE for i in range ( 256 ) : NEW_LINE INDENT sum1 += freq1 [ i ] NEW_LINE sum2 += freq2 [ i ] NEW_LINE DEDENT return max ( sum1 , sum2 ) NEW_LINE
str1 = " geeksforgeeks " NEW_LINE str2 = " geeksforcoder " NEW_LINE print ( ctMinEdits ( str1 , str2 ) ) NEW_LINE
def CountPairs ( a , b , n ) : NEW_LINE
C = [ 0 ] * n NEW_LINE
for i in range ( n ) : NEW_LINE INDENT C [ i ] = a [ i ] + b [ i ] NEW_LINE DEDENT
freqCount = dict ( ) NEW_LINE for i in range ( n ) : NEW_LINE INDENT if C [ i ] in freqCount . keys ( ) : NEW_LINE INDENT freqCount [ C [ i ] ] += 1 NEW_LINE DEDENT else : NEW_LINE INDENT freqCount [ C [ i ] ] = 1 NEW_LINE DEDENT DEDENT
NoOfPairs = 0 NEW_LINE for x in freqCount : NEW_LINE INDENT y = freqCount [ x ] NEW_LINE DEDENT
NoOfPairs = ( NoOfPairs + y * ( y - 1 ) // 2 ) NEW_LINE
print ( NoOfPairs ) NEW_LINE
arr = [ 1 , 4 , 20 , 3 , 10 , 5 ] NEW_LINE brr = [ 9 , 6 , 1 , 7 , 11 , 6 ] NEW_LINE
N = len ( arr ) NEW_LINE
CountPairs ( arr , brr , N ) NEW_LINE
def medianChange ( arr1 , arr2 ) : NEW_LINE INDENT N = len ( arr1 ) NEW_LINE DEDENT
median = [ ] NEW_LINE
if ( N & 1 ) : NEW_LINE INDENT median . append ( arr1 [ N // 2 ] * 1 ) NEW_LINE DEDENT
else : NEW_LINE INDENT median . append ( ( arr1 [ N // 2 ] + arr1 [ ( N - 1 ) // 2 ] ) // 2 ) NEW_LINE DEDENT for x in arr2 : NEW_LINE
it = arr1 . index ( x ) NEW_LINE
arr1 . pop ( it ) NEW_LINE
N -= 1 NEW_LINE
if ( N & 1 ) : NEW_LINE INDENT median . append ( arr1 [ N // 2 ] * 1 ) NEW_LINE DEDENT
else : NEW_LINE INDENT median . append ( ( arr1 [ N // 2 ] + arr1 [ ( N - 1 ) // 2 ] ) // 2 ) NEW_LINE DEDENT
for i in range ( len ( median ) - 1 ) : NEW_LINE INDENT print ( median [ i + 1 ] - median [ i ] , end = ' ▁ ' ) NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE
arr1 = [ 2 , 4 , 6 , 8 , 10 ] NEW_LINE arr2 = [ 4 , 6 ] NEW_LINE
medianChange ( arr1 , arr2 ) NEW_LINE
nfa = 1 NEW_LINE
flag = 0 NEW_LINE
def state1 ( c ) : NEW_LINE INDENT global nfa , flag NEW_LINE DEDENT
if ( c == ' a ' ) : NEW_LINE INDENT nfa = 2 NEW_LINE DEDENT elif ( c == ' b ' or c == ' c ' ) : NEW_LINE INDENT nfa = 1 NEW_LINE DEDENT else : NEW_LINE INDENT flag = 1 NEW_LINE DEDENT
def state2 ( c ) : NEW_LINE INDENT global nfa , flag NEW_LINE DEDENT
if ( c == ' a ' ) : NEW_LINE INDENT nfa = 3 NEW_LINE DEDENT elif ( c == ' b ' or c == ' c ' ) : NEW_LINE INDENT nfa = 2 NEW_LINE DEDENT else : NEW_LINE INDENT flag = 1 NEW_LINE DEDENT
def state3 ( c ) : NEW_LINE INDENT global nfa , flag NEW_LINE DEDENT
if ( c == ' a ' ) : NEW_LINE INDENT nfa = 1 NEW_LINE DEDENT elif ( c == ' b ' or c == ' c ' ) : NEW_LINE INDENT nfa = 3 NEW_LINE DEDENT else : NEW_LINE INDENT flag = 1 NEW_LINE DEDENT
def state4 ( c ) : NEW_LINE INDENT global nfa , flag NEW_LINE DEDENT
if ( c == ' b ' ) : NEW_LINE INDENT nfa = 5 NEW_LINE DEDENT elif ( c == ' a ' or c == ' c ' ) : NEW_LINE INDENT nfa = 4 NEW_LINE DEDENT else : NEW_LINE INDENT flag = 1 NEW_LINE DEDENT
def state5 ( c ) : NEW_LINE INDENT global nfa , flag NEW_LINE DEDENT
if ( c == ' b ' ) : NEW_LINE INDENT nfa = 6 NEW_LINE DEDENT elif ( c == ' a ' or c == ' c ' ) : NEW_LINE INDENT nfa = 5 NEW_LINE DEDENT else : NEW_LINE INDENT flag = 1 NEW_LINE DEDENT
def state6 ( c ) : NEW_LINE INDENT global nfa , flag NEW_LINE DEDENT
if ( c == ' b ' ) : NEW_LINE INDENT nfa = 4 NEW_LINE DEDENT elif ( c == ' a ' or c == ' c ' ) : NEW_LINE INDENT nfa = 6 NEW_LINE DEDENT else : NEW_LINE INDENT flag = 1 NEW_LINE DEDENT
def state7 ( c ) : NEW_LINE INDENT global nfa , flag NEW_LINE DEDENT
if ( c == ' c ' ) : NEW_LINE INDENT nfa = 8 NEW_LINE DEDENT elif ( c == ' b ' or c == ' a ' ) : NEW_LINE INDENT nfa = 7 NEW_LINE DEDENT else : NEW_LINE INDENT flag = 1 NEW_LINE DEDENT
def state8 ( c ) : NEW_LINE INDENT global nfa , flag NEW_LINE DEDENT
if ( c == ' c ' ) : NEW_LINE INDENT nfa = 9 NEW_LINE DEDENT elif ( c == ' b ' or c == ' a ' ) : NEW_LINE INDENT nfa = 8 NEW_LINE DEDENT else : NEW_LINE INDENT flag = 1 NEW_LINE DEDENT
def state9 ( c ) : NEW_LINE INDENT global nfa , flag NEW_LINE DEDENT
if ( c == ' c ' ) : NEW_LINE INDENT nfa = 7 NEW_LINE DEDENT elif ( c == ' b ' or c == ' a ' ) : NEW_LINE INDENT nfa = 9 NEW_LINE DEDENT else : NEW_LINE INDENT flag = 1 NEW_LINE DEDENT
def checkA ( s , x ) : NEW_LINE INDENT global nfa , flag NEW_LINE for i in range ( x ) : NEW_LINE INDENT if ( nfa == 1 ) : NEW_LINE INDENT state1 ( s [ i ] ) NEW_LINE DEDENT elif ( nfa == 2 ) : NEW_LINE INDENT state2 ( s [ i ] ) NEW_LINE DEDENT elif ( nfa == 3 ) : NEW_LINE INDENT state3 ( s [ i ] ) NEW_LINE DEDENT DEDENT if ( nfa == 1 ) : NEW_LINE INDENT return True NEW_LINE DEDENT else : NEW_LINE INDENT nfa = 4 NEW_LINE DEDENT DEDENT
def checkB ( s , x ) : NEW_LINE INDENT global nfa , flag NEW_LINE for i in range ( x ) : NEW_LINE INDENT if ( nfa == 4 ) : NEW_LINE INDENT state4 ( s [ i ] ) NEW_LINE DEDENT elif ( nfa == 5 ) : NEW_LINE INDENT state5 ( s [ i ] ) NEW_LINE DEDENT elif ( nfa == 6 ) : NEW_LINE INDENT state6 ( s [ i ] ) NEW_LINE DEDENT DEDENT if ( nfa == 4 ) : NEW_LINE INDENT return True NEW_LINE DEDENT else : NEW_LINE INDENT nfa = 7 NEW_LINE DEDENT DEDENT
def checkC ( s , x ) : NEW_LINE INDENT global nfa , flag NEW_LINE for i in range ( x ) : NEW_LINE INDENT if ( nfa == 7 ) : NEW_LINE INDENT state7 ( s [ i ] ) NEW_LINE DEDENT elif ( nfa == 8 ) : NEW_LINE INDENT state8 ( s [ i ] ) NEW_LINE DEDENT elif ( nfa == 9 ) : NEW_LINE INDENT state9 ( s [ i ] ) NEW_LINE DEDENT DEDENT if ( nfa == 7 ) : NEW_LINE INDENT return True NEW_LINE DEDENT DEDENT
s = " bbbca " NEW_LINE x = 5 NEW_LINE
if ( checkA ( s , x ) or checkB ( s , x ) or checkC ( s , x ) ) : NEW_LINE INDENT print ( " ACCEPTED " ) NEW_LINE DEDENT else : NEW_LINE INDENT if ( flag == 0 ) : NEW_LINE INDENT print ( " NOT ▁ ACCEPTED " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " INPUT ▁ OUT ▁ OF ▁ DICTIONARY . " ) NEW_LINE DEDENT DEDENT
def getPositionCount ( a , n ) : NEW_LINE
count = 1 ; NEW_LINE
min = a [ 0 ] ; NEW_LINE
for i in range ( 1 , n ) : NEW_LINE
if ( a [ i ] <= min ) : NEW_LINE
min = a [ i ] ; NEW_LINE
count += 1 ; NEW_LINE return count ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = [ 5 , 4 , 6 , 1 , 3 , 1 ] ; NEW_LINE n = len ( a ) ; NEW_LINE print ( getPositionCount ( a , n ) ) ; NEW_LINE DEDENT
def maxSum ( arr , n , k ) : NEW_LINE
if ( n < k ) : NEW_LINE INDENT return - 1 ; NEW_LINE DEDENT
res = 0 ; NEW_LINE for i in range ( k ) : NEW_LINE INDENT res += arr [ i ] ; NEW_LINE DEDENT
curr_sum = res ; NEW_LINE for i in range ( k , n ) : NEW_LINE INDENT curr_sum += arr [ i ] - arr [ i - k ] ; NEW_LINE res = max ( res , curr_sum ) ; NEW_LINE DEDENT return res ; NEW_LINE
def solve ( arr , n , k ) : NEW_LINE INDENT max_len = 0 ; l = 0 ; r = n ; NEW_LINE DEDENT
while ( l <= r ) : NEW_LINE INDENT m = ( l + r ) // 2 ; NEW_LINE DEDENT
if ( maxSum ( arr , n , m ) > k ) : NEW_LINE INDENT r = m - 1 ; NEW_LINE DEDENT else : NEW_LINE INDENT l = m + 1 ; NEW_LINE DEDENT
max_len = m ; NEW_LINE return max_len ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 1 , 2 , 3 , 4 , 5 ] ; NEW_LINE n = len ( arr ) ; NEW_LINE k = 10 ; NEW_LINE print ( solve ( arr , n , k ) ) ; NEW_LINE DEDENT
MAX = 100001 NEW_LINE ROW = 10 NEW_LINE COL = 3 NEW_LINE indices = [ 0 ] * MAX NEW_LINE
test = [ [ 2 , 3 , 6 ] , [ 2 , 4 , 4 ] , [ 2 , 6 , 3 ] , [ 3 , 2 , 6 ] , [ 3 , 3 , 3 ] , [ 3 , 6 , 2 ] , [ 4 , 2 , 4 ] , [ 4 , 4 , 2 ] , [ 6 , 2 , 3 ] , [ 6 , 3 , 2 ] ] NEW_LINE
def find_triplet ( array , n ) : NEW_LINE INDENT answer = 0 NEW_LINE for i in range ( MAX ) : NEW_LINE INDENT indices [ i ] = [ ] NEW_LINE DEDENT DEDENT
for i in range ( n ) : NEW_LINE INDENT indices [ array [ i ] ] . append ( i ) NEW_LINE DEDENT for i in range ( n ) : NEW_LINE INDENT y = array [ i ] NEW_LINE for j in range ( ROW ) : NEW_LINE INDENT s = test [ j ] [ 1 ] * y NEW_LINE DEDENT DEDENT
if s % test [ j ] [ 0 ] != 0 : NEW_LINE INDENT continue NEW_LINE DEDENT if s % test [ j ] [ 2 ] != 0 : NEW_LINE INDENT continue NEW_LINE DEDENT x = s // test [ j ] [ 0 ] NEW_LINE z = s // test [ j ] [ 2 ] NEW_LINE if x > MAX or z > MAX : NEW_LINE INDENT continue NEW_LINE DEDENT l = 0 NEW_LINE r = len ( indices [ x ] ) - 1 NEW_LINE first = - 1 NEW_LINE
while l <= r : NEW_LINE INDENT m = ( l + r ) // 2 NEW_LINE if indices [ x ] [ m ] < i : NEW_LINE INDENT first = m NEW_LINE l = m + 1 NEW_LINE DEDENT else : NEW_LINE INDENT r = m - 1 NEW_LINE DEDENT DEDENT l = 0 NEW_LINE r = len ( indices [ z ] ) - 1 NEW_LINE third = - 1 NEW_LINE
while l <= r : NEW_LINE INDENT m = ( l + r ) // 2 NEW_LINE if indices [ z ] [ m ] > i : NEW_LINE INDENT third = m NEW_LINE r = m - 1 NEW_LINE DEDENT else : NEW_LINE INDENT l = m + 1 NEW_LINE DEDENT DEDENT if first != - 1 and third != - 1 : NEW_LINE
answer += ( first + 1 ) * ( len ( indices [ z ] ) - third ) NEW_LINE return answer NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT array = [ 2 , 4 , 5 , 6 , 7 ] NEW_LINE n = len ( array ) NEW_LINE print ( find_triplet ( array , n ) ) NEW_LINE DEDENT
def distinct ( arr ) : NEW_LINE INDENT count = 0 NEW_LINE DEDENT
if len ( arr ) == 1 : NEW_LINE INDENT return 1 NEW_LINE DEDENT for i in range ( 0 , len ( arr ) - 1 ) : NEW_LINE
if ( i == 0 ) : NEW_LINE INDENT if ( arr [ i ] != arr [ i + 1 ] ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT DEDENT
elif ( i > 0 & i < len ( arr ) - 1 ) : NEW_LINE INDENT if ( arr [ i ] != arr [ i + 1 ] or arr [ i ] != arr [ i - 1 ] ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT DEDENT
if ( arr [ len ( arr ) - 1 ] != arr [ len ( arr ) - 2 ] ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT return count NEW_LINE
arr = [ 0 , 0 , 0 , 0 , 0 , 1 , 0 ] NEW_LINE print ( distinct ( arr ) ) NEW_LINE
def isSorted ( arr , N ) : NEW_LINE
for i in range ( 1 , N ) : NEW_LINE INDENT if ( arr [ i ] [ 0 ] > arr [ i - 1 ] [ 0 ] ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT
return True NEW_LINE
def isPossibleToSort ( arr , N ) : NEW_LINE
group = arr [ 0 ] [ 1 ] NEW_LINE
for i in range ( 1 , N ) : NEW_LINE
if ( arr [ i ] [ 1 ] != group ) : NEW_LINE INDENT return " Yes " NEW_LINE DEDENT
if ( isSorted ( arr , N ) ) : NEW_LINE INDENT return " Yes " NEW_LINE DEDENT else : NEW_LINE INDENT return " No " NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ [ 340000 , 2 ] , [ 45000 , 1 ] , [ 30000 , 2 ] , [ 50000 , 4 ] ] NEW_LINE N = len ( arr ) NEW_LINE print ( isPossibleToSort ( arr , N ) ) NEW_LINE DEDENT
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT DEDENT sum = 0 NEW_LINE total_sum = 0 NEW_LINE mod = 1000000007 NEW_LINE
def getAlphaScore ( node ) : NEW_LINE INDENT global sum NEW_LINE global total_sum NEW_LINE DEDENT
if node . left != None : NEW_LINE INDENT getAlphaScore ( node . left ) NEW_LINE DEDENT
sum = ( sum + node . data ) % mod NEW_LINE
total_sum = ( total_sum + sum ) % mod NEW_LINE
if node . right != None : NEW_LINE INDENT getAlphaScore ( node . right ) NEW_LINE DEDENT
return total_sum NEW_LINE
def constructBST ( arr , start , end , root ) : NEW_LINE INDENT if start > end : NEW_LINE INDENT return None NEW_LINE DEDENT mid = ( start + end ) // 2 NEW_LINE DEDENT
if root == None : NEW_LINE INDENT root = Node ( arr [ mid ] ) NEW_LINE DEDENT
root . left = constructBST ( arr , start , mid - 1 , root . left ) NEW_LINE
root . right = constructBST ( arr , mid + 1 , end , root . right ) NEW_LINE
return root NEW_LINE
arr = [ 10 , 11 , 12 ] NEW_LINE length = len ( arr ) NEW_LINE
arr . sort ( ) NEW_LINE root = None NEW_LINE
root = constructBST ( arr , 0 , length - 1 , root ) NEW_LINE print ( getAlphaScore ( root ) ) NEW_LINE
def sortByFreq ( arr , n ) : NEW_LINE
maxE = - 1 ; NEW_LINE
for i in range ( n ) : NEW_LINE INDENT maxE = max ( maxE , arr [ i ] ) NEW_LINE DEDENT
freq = [ 0 ] * ( maxE + 1 ) ; NEW_LINE
for i in range ( n ) : NEW_LINE INDENT freq [ arr [ i ] ] += 1 ; NEW_LINE DEDENT
cnt = 0 ; NEW_LINE
for i in range ( maxE + 1 ) : NEW_LINE
if ( freq [ i ] > 0 ) : NEW_LINE INDENT value = 100000 - i ; NEW_LINE arr [ cnt ] = 100000 * freq [ i ] + value ; NEW_LINE cnt += 1 ; NEW_LINE DEDENT
return cnt ; NEW_LINE
def printSortedArray ( arr , cnt ) : NEW_LINE
for i in range ( cnt ) : NEW_LINE
frequency = arr [ i ] / 100000 ; NEW_LINE
value = 100000 - ( arr [ i ] % 100000 ) ; NEW_LINE
for j in range ( int ( frequency ) ) : NEW_LINE INDENT print ( value , end = " ▁ " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 4 , 4 , 5 , 6 , 4 , 2 , 2 , 8 , 5 ] NEW_LINE DEDENT
n = len ( arr ) NEW_LINE
cnt = sortByFreq ( arr , n ) ; NEW_LINE
arr . sort ( reverse = True ) NEW_LINE
printSortedArray ( arr , cnt ) ; NEW_LINE
def checkRectangles ( arr , n ) : NEW_LINE INDENT ans = True NEW_LINE DEDENT
arr . sort ( ) NEW_LINE
area = arr [ 0 ] * arr [ 4 * n - 1 ] NEW_LINE
for i in range ( 0 , 2 * n , 2 ) : NEW_LINE INDENT if ( arr [ i ] != arr [ i + 1 ] or arr [ 4 * n - i - 1 ] != arr [ 4 * n - i - 2 ] or arr [ i ] * arr [ 4 * n - i - 1 ] != area ) : NEW_LINE DEDENT
ans = False NEW_LINE break NEW_LINE
if ( ans ) : NEW_LINE INDENT return True NEW_LINE DEDENT return False NEW_LINE
arr = [ 1 , 8 , 2 , 1 , 2 , 4 , 4 , 8 ] NEW_LINE n = 2 NEW_LINE if ( checkRectangles ( arr , n ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
def cntElements ( arr , n ) : NEW_LINE
copy_arr = [ 0 ] * n NEW_LINE
for i in range ( n ) : NEW_LINE INDENT copy_arr [ i ] = arr [ i ] NEW_LINE DEDENT
count = 0 NEW_LINE
arr . sort ( ) NEW_LINE for i in range ( n ) : NEW_LINE
if ( arr [ i ] != copy_arr [ i ] ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT return count NEW_LINE
arr = [ 1 , 2 , 6 , 2 , 4 , 5 ] NEW_LINE n = len ( arr ) NEW_LINE print ( cntElements ( arr , n ) ) NEW_LINE
def findPairs ( arr , n , k , d ) : NEW_LINE
if ( n < 2 * k ) : NEW_LINE INDENT print ( " - 1" ) NEW_LINE return NEW_LINE DEDENT
pairs = [ ] NEW_LINE
arr = sorted ( arr ) NEW_LINE
for i in range ( k ) : NEW_LINE
if ( arr [ n - k + i ] - arr [ i ] >= d ) : NEW_LINE
pairs . append ( [ arr [ i ] , arr [ n - k + i ] ] ) NEW_LINE
if ( len ( pairs ) < k ) : NEW_LINE INDENT print ( " - 1" ) NEW_LINE return NEW_LINE DEDENT
for v in pairs : NEW_LINE INDENT print ( " ( " , v [ 0 ] , " , ▁ " , v [ 1 ] , " ) " ) NEW_LINE DEDENT
arr = [ 4 , 6 , 10 , 23 , 14 , 7 , 2 , 20 , 9 ] NEW_LINE n = len ( arr ) NEW_LINE k = 4 NEW_LINE d = 3 NEW_LINE findPairs ( arr , n , k , d ) NEW_LINE
def pairs_count ( arr , n , sum ) : NEW_LINE
ans = 0 NEW_LINE
arr = sorted ( arr ) NEW_LINE
i , j = 0 , n - 1 NEW_LINE while ( i < j ) : NEW_LINE
if ( arr [ i ] + arr [ j ] < sum ) : NEW_LINE INDENT i += 1 NEW_LINE DEDENT
elif ( arr [ i ] + arr [ j ] > sum ) : NEW_LINE INDENT j -= 1 NEW_LINE DEDENT
else : NEW_LINE
x = arr [ i ] NEW_LINE xx = i NEW_LINE while ( i < j and arr [ i ] == x ) : NEW_LINE INDENT i += 1 NEW_LINE DEDENT
y = arr [ j ] NEW_LINE yy = j NEW_LINE while ( j >= i and arr [ j ] == y ) : NEW_LINE INDENT j -= 1 NEW_LINE DEDENT
if ( x == y ) : NEW_LINE INDENT temp = i - xx + yy - j - 1 NEW_LINE ans += ( temp * ( temp + 1 ) ) // 2 NEW_LINE DEDENT else : NEW_LINE INDENT ans += ( i - xx ) * ( yy - j ) NEW_LINE DEDENT
return ans NEW_LINE
arr = [ 1 , 5 , 7 , 5 , - 1 ] NEW_LINE n = len ( arr ) NEW_LINE sum = 6 NEW_LINE print ( pairs_count ( arr , n , sum ) ) NEW_LINE
import sys NEW_LINE def check ( str ) : NEW_LINE INDENT min = sys . maxsize NEW_LINE max = - sys . maxsize - 1 NEW_LINE sum = 0 NEW_LINE DEDENT
for i in range ( len ( str ) ) : NEW_LINE
ascii = str [ i ] NEW_LINE
if ( ord ( ascii ) < 96 or ord ( ascii ) > 122 ) : NEW_LINE INDENT return False NEW_LINE DEDENT
sum += ord ( ascii ) NEW_LINE
if ( min > ord ( ascii ) ) : NEW_LINE INDENT min = ord ( ascii ) NEW_LINE DEDENT
if ( max < ord ( ascii ) ) : NEW_LINE INDENT max = ord ( ascii ) NEW_LINE DEDENT
min -= 1 NEW_LINE
eSum = ( ( ( max * ( max + 1 ) ) // 2 ) - ( ( min * ( min + 1 ) ) // 2 ) ) NEW_LINE
return sum == eSum NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
str = " dcef " NEW_LINE if ( check ( str ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
str1 = " xyza " NEW_LINE if ( check ( str1 ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
def findKth ( arr , n , k ) : NEW_LINE INDENT missing = dict ( ) NEW_LINE count = 0 NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT missing [ arr [ i ] ] = 1 NEW_LINE DEDENT
maxm = max ( arr ) NEW_LINE minm = min ( arr ) NEW_LINE
for i in range ( minm + 1 , maxm ) : NEW_LINE
if ( i not in missing . keys ( ) ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT
if ( count == k ) : NEW_LINE INDENT return i NEW_LINE DEDENT
return - 1 NEW_LINE
arr = [ 2 , 10 , 9 , 4 ] NEW_LINE n = len ( arr ) NEW_LINE k = 5 NEW_LINE print ( findKth ( arr , n , k ) ) NEW_LINE
import math NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . next = None NEW_LINE DEDENT DEDENT
def sortList ( head ) : NEW_LINE INDENT startVal = 1 NEW_LINE while ( head != None ) : NEW_LINE INDENT head . data = startVal NEW_LINE startVal = startVal + 1 NEW_LINE head = head . next NEW_LINE DEDENT DEDENT
def push ( head_ref , new_data ) : NEW_LINE
new_node = Node ( new_data ) NEW_LINE
new_node . data = new_data NEW_LINE
new_node . next = head_ref NEW_LINE
head_ref = new_node NEW_LINE return head_ref NEW_LINE
def prList ( node ) : NEW_LINE INDENT while ( node != None ) : NEW_LINE INDENT print ( node . data , end = " ▁ " ) NEW_LINE node = node . next NEW_LINE DEDENT DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT head = None NEW_LINE DEDENT
head = push ( head , 2 ) NEW_LINE head = push ( head , 1 ) NEW_LINE head = push ( head , 6 ) NEW_LINE head = push ( head , 4 ) NEW_LINE head = push ( head , 5 ) NEW_LINE head = push ( head , 3 ) NEW_LINE sortList ( head ) NEW_LINE prList ( head ) NEW_LINE
class Node : NEW_LINE INDENT def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . next = None NEW_LINE DEDENT DEDENT
def isSortedDesc ( head ) : NEW_LINE
if ( head == None or head . next == None ) : NEW_LINE INDENT return True NEW_LINE DEDENT
return ( head . data > head . next . data and isSortedDesc ( head . next ) ) NEW_LINE def newNode ( data ) : NEW_LINE INDENT temp = Node ( data ) NEW_LINE return temp NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT head = newNode ( 7 ) NEW_LINE head . next = newNode ( 5 ) NEW_LINE head . next . next = newNode ( 4 ) NEW_LINE head . next . next . next = newNode ( 3 ) NEW_LINE if isSortedDesc ( head ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT DEDENT
def minSum ( arr , n ) : NEW_LINE
evenArr = [ ] NEW_LINE oddArr = [ ] NEW_LINE
arr . sort ( ) NEW_LINE
for i in range ( n ) : NEW_LINE INDENT if ( i < n // 2 ) : NEW_LINE INDENT oddArr . append ( arr [ i ] ) NEW_LINE DEDENT else : NEW_LINE INDENT evenArr . append ( arr [ i ] ) NEW_LINE DEDENT DEDENT
evenArr . sort ( reverse = True ) NEW_LINE
i = 0 NEW_LINE sum = 0 NEW_LINE for j in range ( len ( evenArr ) ) : NEW_LINE INDENT arr [ i ] = evenArr [ j ] NEW_LINE i += 1 NEW_LINE arr [ i ] = oddArr [ j ] NEW_LINE i += 1 NEW_LINE sum += evenArr [ j ] * oddArr [ j ] NEW_LINE DEDENT return sum NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 1 , 5 , 8 , 9 , 6 , 7 , 3 , 4 , 2 , 0 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Minimum ▁ required ▁ sum ▁ = " , minSum ( arr , n ) ) NEW_LINE print ( " Sorted ▁ array ▁ in ▁ required ▁ format ▁ : ▁ " , end = " " ) NEW_LINE for i in range ( n ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT DEDENT
def minTime ( word ) : NEW_LINE INDENT ans = 0 NEW_LINE DEDENT
curr = 0 NEW_LINE for i in range ( len ( word ) ) : NEW_LINE
k = ord ( word [ i ] ) - 97 NEW_LINE
a = abs ( curr - k ) NEW_LINE
b = 26 - abs ( curr - k ) NEW_LINE
ans += min ( a , b ) NEW_LINE
ans += 1 NEW_LINE curr = ord ( word [ i ] ) - 97 NEW_LINE
print ( ans ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
str = " zjpc " NEW_LINE
minTime ( str ) NEW_LINE
def reduceToOne ( N ) : NEW_LINE
cnt = 0 NEW_LINE while ( N != 1 ) : NEW_LINE
if ( N == 2 or ( N % 2 == 1 ) ) : NEW_LINE
N = N - 1 NEW_LINE
cnt += 1 NEW_LINE
elif ( N % 2 == 0 ) : NEW_LINE
N = N / ( N / 2 ) NEW_LINE
cnt += 1 NEW_LINE
return cnt NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 35 NEW_LINE print ( reduceToOne ( N ) ) NEW_LINE DEDENT
def maxDiamonds ( A , N , K ) : NEW_LINE
pq = [ ] NEW_LINE
for i in range ( N ) : NEW_LINE INDENT pq . append ( A [ i ] ) NEW_LINE DEDENT pq . sort ( ) NEW_LINE
ans = 0 NEW_LINE
while ( len ( pq ) > 0 and K > 0 ) : NEW_LINE INDENT pq . sort ( ) NEW_LINE DEDENT
top = pq [ len ( pq ) - 1 ] NEW_LINE
pq = pq [ 0 : len ( pq ) - 1 ] NEW_LINE
ans += top NEW_LINE
top = top // 2 ; NEW_LINE pq . append ( top ) NEW_LINE K -= 1 NEW_LINE
print ( ans ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT A = [ 2 , 1 , 7 , 4 , 2 ] NEW_LINE K = 3 NEW_LINE N = len ( A ) NEW_LINE maxDiamonds ( A , N , K ) NEW_LINE DEDENT
def MinimumCost ( A , B , N ) : NEW_LINE
totalCost = 0 NEW_LINE
for i in range ( N ) : NEW_LINE
mod_A = B [ i ] % A [ i ] NEW_LINE totalCost_A = min ( mod_A , A [ i ] - mod_A ) NEW_LINE
mod_B = A [ i ] % B [ i ] NEW_LINE totalCost_B = min ( mod_B , B [ i ] - mod_B ) NEW_LINE
totalCost += min ( totalCost_A , totalCost_B ) NEW_LINE
return totalCost NEW_LINE
A = [ 3 , 6 , 3 ] NEW_LINE B = [ 4 , 8 , 13 ] NEW_LINE N = len ( A ) NEW_LINE print ( MinimumCost ( A , B , N ) ) NEW_LINE
def printLargestDivisible ( arr , N ) : NEW_LINE INDENT count0 = 0 ; count7 = 0 ; NEW_LINE for i in range ( N ) : NEW_LINE DEDENT
if ( arr [ i ] == 0 ) : NEW_LINE INDENT count0 += 1 ; NEW_LINE DEDENT else : NEW_LINE INDENT count7 += 1 ; NEW_LINE DEDENT
if ( count7 % 50 == 0 ) : NEW_LINE INDENT while ( count7 ) : NEW_LINE INDENT count7 -= 1 ; NEW_LINE print ( 7 , end = " " ) ; NEW_LINE DEDENT while ( count0 ) : NEW_LINE INDENT count0 -= 1 ; NEW_LINE print ( count0 , end = " " ) ; NEW_LINE DEDENT DEDENT
elif ( count7 < 5 ) : NEW_LINE INDENT if ( count0 == 0 ) : NEW_LINE INDENT print ( " No " , end = " " ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( "0" , end = " " ) ; NEW_LINE DEDENT DEDENT
else : NEW_LINE
count7 = count7 - count7 % 5 ; NEW_LINE while ( count7 ) : NEW_LINE INDENT count7 -= 1 ; NEW_LINE print ( 7 , end = " " ) ; NEW_LINE DEDENT while ( count0 ) : NEW_LINE INDENT count0 -= 1 ; NEW_LINE print ( 0 , end = " " ) ; NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE
arr = [ 0 , 7 , 0 , 7 , 7 , 7 , 7 , 0 , 0 , 0 , 0 , 0 , 0 , 7 , 7 , 7 ] ; NEW_LINE
N = len ( arr ) ; NEW_LINE printLargestDivisible ( arr , N ) ; NEW_LINE
def findMaxValByRearrArr ( arr , N ) : NEW_LINE
arr . sort ( ) NEW_LINE
res = 0 NEW_LINE
while ( True ) : NEW_LINE
Sum = 0 NEW_LINE
for i in range ( N ) : NEW_LINE
Sum += __gcd ( i + 1 , arr [ i ] ) NEW_LINE
res = max ( res , Sum ) NEW_LINE if ( not next_permutation ( arr ) ) : NEW_LINE INDENT break NEW_LINE DEDENT return res NEW_LINE def __gcd ( a , b ) : NEW_LINE if b == 0 : NEW_LINE return a NEW_LINE else : NEW_LINE return __gcd ( b , a % b ) NEW_LINE def next_permutation ( p ) : NEW_LINE for a in range ( len ( p ) - 2 , - 1 , - 1 ) : NEW_LINE if ( p [ a ] < p [ a + 1 ] ) : NEW_LINE INDENT b = len ( p ) - 1 NEW_LINE while True : NEW_LINE INDENT if ( p [ b ] > p [ a ] ) : NEW_LINE INDENT t = p [ a ] NEW_LINE p [ a ] = p [ b ] NEW_LINE p [ b ] = t NEW_LINE a += 1 NEW_LINE b = len ( p ) - 1 NEW_LINE while a < b : NEW_LINE INDENT t = p [ a ] NEW_LINE p [ a ] = p [ b ] NEW_LINE p [ b ] = t NEW_LINE a += 1 NEW_LINE b -= 1 NEW_LINE DEDENT return True NEW_LINE DEDENT b -= 1 NEW_LINE DEDENT DEDENT return False NEW_LINE
arr = [ 3 , 2 , 1 ] NEW_LINE N = len ( arr ) NEW_LINE print ( findMaxValByRearrArr ( arr , N ) ) NEW_LINE
def min_elements ( arr , N ) : NEW_LINE
mp = { } ; NEW_LINE
for i in range ( N ) : NEW_LINE
if arr [ i ] in mp : NEW_LINE INDENT mp [ arr [ i ] ] += 1 ; NEW_LINE DEDENT else : NEW_LINE INDENT mp [ arr [ i ] ] = 1 ; NEW_LINE DEDENT
cntMinRem = 0 ; NEW_LINE
for it in mp : NEW_LINE
i = it ; NEW_LINE
if ( mp [ i ] < i ) : NEW_LINE
cntMinRem += mp [ i ] ; NEW_LINE
elif ( mp [ i ] > i ) : NEW_LINE
cntMinRem += ( mp [ i ] - i ) ; NEW_LINE return cntMinRem ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 2 , 4 , 1 , 4 , 2 ] ; NEW_LINE N = len ( arr ) ; NEW_LINE print ( min_elements ( arr , N ) ) ; NEW_LINE DEDENT
def CheckAllarrayEqual ( arr , N ) : NEW_LINE
if ( N == 1 ) : NEW_LINE INDENT return True NEW_LINE DEDENT
totalSum = arr [ 0 ] NEW_LINE
secMax = - 10 ** 19 NEW_LINE
Max = arr [ 0 ] NEW_LINE
for i in range ( 1 , N ) : NEW_LINE INDENT if ( arr [ i ] >= Max ) : NEW_LINE DEDENT
secMax = Max NEW_LINE
Max = arr [ i ] NEW_LINE elif ( arr [ i ] > secMax ) : NEW_LINE
secMax = arr [ i ] NEW_LINE
totalSum += arr [ i ] NEW_LINE
if ( ( secMax * ( N - 1 ) ) > totalSum ) : NEW_LINE INDENT return False NEW_LINE DEDENT
if ( totalSum % ( N - 1 ) ) : NEW_LINE INDENT return False NEW_LINE DEDENT return True NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 6 , 2 , 2 , 2 ] NEW_LINE N = len ( arr ) NEW_LINE if ( CheckAllarrayEqual ( arr , N ) ) : NEW_LINE INDENT print ( " YES " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NO " ) NEW_LINE DEDENT DEDENT
def Remove_one_element ( arr , n ) : NEW_LINE
post_odd = 0 NEW_LINE post_even = 0 NEW_LINE
curr_odd = 0 NEW_LINE curr_even = 0 NEW_LINE
res = 0 NEW_LINE
for i in range ( n - 1 , - 1 , - 1 ) : NEW_LINE
if ( i % 2 ) : NEW_LINE INDENT post_odd ^= arr [ i ] NEW_LINE DEDENT
else : NEW_LINE INDENT post_even ^= arr [ i ] NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE
if ( i % 2 ) : NEW_LINE INDENT post_odd ^= arr [ i ] NEW_LINE DEDENT
else : NEW_LINE INDENT post_even ^= arr [ i ] NEW_LINE DEDENT
X = curr_odd ^ post_even NEW_LINE
Y = curr_even ^ post_odd NEW_LINE
if ( X == Y ) : NEW_LINE INDENT res += 1 NEW_LINE DEDENT
if ( i % 2 ) : NEW_LINE INDENT curr_odd ^= arr [ i ] NEW_LINE DEDENT
else : NEW_LINE INDENT curr_even ^= arr [ i ] NEW_LINE DEDENT
print ( res ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
arr = [ 1 , 0 , 1 , 0 , 1 ] NEW_LINE
N = len ( arr ) NEW_LINE
Remove_one_element ( arr , N ) NEW_LINE
def cntIndexesToMakeBalance ( arr , n ) : NEW_LINE
if ( n == 1 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT
if ( n == 2 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
sumEven = 0 NEW_LINE
sumOdd = 0 NEW_LINE
for i in range ( n ) : NEW_LINE
if ( i % 2 == 0 ) : NEW_LINE
sumEven += arr [ i ] NEW_LINE
else : NEW_LINE
sumOdd += arr [ i ] NEW_LINE
currOdd = 0 NEW_LINE
currEven = arr [ 0 ] NEW_LINE
res = 0 NEW_LINE
newEvenSum = 0 NEW_LINE
newOddSum = 0 NEW_LINE
for i in range ( 1 , n - 1 ) : NEW_LINE
if ( i % 2 ) : NEW_LINE
currOdd += arr [ i ] NEW_LINE
newEvenSum = ( currEven + sumOdd - currOdd ) NEW_LINE
newOddSum = ( currOdd + sumEven - currEven - arr [ i ] ) NEW_LINE
else : NEW_LINE
currEven += arr [ i ] NEW_LINE
newOddSum = ( currOdd + sumEven - currEven ) NEW_LINE
newEvenSum = ( currEven + sumOdd - currOdd - arr [ i ] ) NEW_LINE
if ( newEvenSum == newOddSum ) : NEW_LINE
res += 1 NEW_LINE
if ( sumOdd == sumEven - arr [ 0 ] ) : NEW_LINE
res += 1 NEW_LINE
if ( n % 2 == 1 ) : NEW_LINE
if ( sumOdd == sumEven - arr [ n - 1 ] ) : NEW_LINE
res += 1 NEW_LINE
else : NEW_LINE
if ( sumEven == sumOdd - arr [ n - 1 ] ) : NEW_LINE
res += 1 NEW_LINE return res NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 1 , 1 , 1 ] NEW_LINE n = len ( arr ) NEW_LINE print ( cntIndexesToMakeBalance ( arr , n ) ) NEW_LINE DEDENT
def findNums ( X , Y ) : NEW_LINE
A = 0 ; NEW_LINE B = 0 ; NEW_LINE
if ( X < Y ) : NEW_LINE INDENT A = - 1 ; NEW_LINE B = - 1 ; NEW_LINE DEDENT
elif ( ( ( abs ( X - Y ) ) & 1 ) != 0 ) : NEW_LINE INDENT A = - 1 ; NEW_LINE B = - 1 ; NEW_LINE DEDENT
elif ( X == Y ) : NEW_LINE INDENT A = 0 ; NEW_LINE B = Y ; NEW_LINE DEDENT
else : NEW_LINE
A = ( X - Y ) // 2 ; NEW_LINE
if ( ( A & Y ) == 0 ) : NEW_LINE
B = ( A + Y ) ; NEW_LINE
else : NEW_LINE INDENT A = - 1 ; NEW_LINE B = - 1 ; NEW_LINE DEDENT
print A ; NEW_LINE print B ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
X = 17 ; NEW_LINE Y = 13 ; NEW_LINE
findNums ( X , Y ) ; NEW_LINE
def checkCount ( A , Q , q ) : NEW_LINE
for i in range ( q ) : NEW_LINE INDENT L = Q [ i ] [ 0 ] NEW_LINE R = Q [ i ] [ 1 ] NEW_LINE DEDENT
L -= 1 NEW_LINE R -= 1 NEW_LINE
if ( ( A [ L ] < A [ L + 1 ] ) != ( A [ R - 1 ] < A [ R ] ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 11 , 13 , 12 , 14 ] NEW_LINE Q = [ [ 1 , 4 ] , [ 2 , 4 ] ] NEW_LINE q = len ( Q ) NEW_LINE checkCount ( arr , Q , q ) NEW_LINE DEDENT
def pairProductMean ( arr , N ) : NEW_LINE
pairArray = [ ] ; NEW_LINE
for i in range ( N ) : NEW_LINE INDENT for j in range ( i + 1 , N ) : NEW_LINE INDENT pairProduct = arr [ i ] * arr [ j ] ; NEW_LINE DEDENT DEDENT
pairArray . append ( pairProduct ) ; NEW_LINE
length = len ( pairArray ) ; NEW_LINE
sum = 0 ; NEW_LINE for i in range ( length ) : NEW_LINE INDENT sum += pairArray [ i ] ; NEW_LINE DEDENT
mean = 0 ; NEW_LINE
if ( length != 0 ) : NEW_LINE INDENT mean = sum / length ; NEW_LINE DEDENT else : NEW_LINE INDENT mean = 0 ; NEW_LINE DEDENT
return mean ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
arr = [ 1 , 2 , 4 , 8 ] ; NEW_LINE N = len ( arr ) ; NEW_LINE
print ( " { 0 : . 2f } " . format ( pairProductMean ( arr , N ) ) ) NEW_LINE
def findPlayer ( str , n ) : NEW_LINE
move_first = 0 NEW_LINE
move_sec = 0 NEW_LINE
for i in range ( n ) : NEW_LINE
if ( str [ i ] [ 0 ] == str [ i ] [ len ( str [ i ] ) - 1 ] ) : NEW_LINE
' NEW_LINE INDENT if ( str [ i ] [ 0 ] == 48 ) : NEW_LINE INDENT move_first += 1 NEW_LINE DEDENT else : NEW_LINE INDENT move_sec += 1 NEW_LINE DEDENT DEDENT
if ( move_first <= move_sec ) : NEW_LINE INDENT print ( " Player ▁ 2 ▁ wins " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Player ▁ 1 ▁ wins " ) NEW_LINE DEDENT
str = [ "010" , "101" ] NEW_LINE N = len ( str ) NEW_LINE
findPlayer ( str , N ) NEW_LINE
def find_next ( n , k ) : NEW_LINE
M = n + 1 ; NEW_LINE while ( True ) : NEW_LINE
if ( ( M & ( 1 << k ) ) > 0 ) : NEW_LINE INDENT break ; NEW_LINE DEDENT
M += 1 ; NEW_LINE
return M ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
N = 15 ; K = 2 ; NEW_LINE
print ( find_next ( N , K ) ) ; NEW_LINE
def find_next ( n , k ) : NEW_LINE
ans = 0 NEW_LINE
if ( ( n & ( 1 << k ) ) == 0 ) : NEW_LINE INDENT cur = 0 NEW_LINE DEDENT
for i in range ( k ) : NEW_LINE
if ( n & ( 1 << i ) ) : NEW_LINE INDENT cur += 1 << i NEW_LINE DEDENT
ans = n - cur + ( 1 << k ) NEW_LINE
else : NEW_LINE INDENT first_unset_bit , cur = - 1 , 0 NEW_LINE for i in range ( 64 ) : NEW_LINE DEDENT
if ( ( n & ( 1 << i ) ) == 0 ) : NEW_LINE INDENT first_unset_bit = i NEW_LINE break NEW_LINE DEDENT
else : NEW_LINE INDENT cur += ( 1 << i ) NEW_LINE DEDENT
ans = n - cur + ( 1 << first_unset_bit ) NEW_LINE
if ( ( ans & ( 1 << k ) ) == 0 ) : NEW_LINE INDENT ans += ( 1 << k ) NEW_LINE DEDENT
return ans NEW_LINE
N , K = 15 , 2 NEW_LINE
print ( find_next ( N , K ) ) NEW_LINE
def largestString ( num , k ) : NEW_LINE
ans = [ ] NEW_LINE for i in range ( len ( num ) ) : NEW_LINE
while ( len ( ans ) and ans [ - 1 ] < num [ i ] and k > 0 ) : NEW_LINE
ans . pop ( ) NEW_LINE
k -= 1 NEW_LINE
ans . append ( num [ i ] ) NEW_LINE
while ( len ( ans ) and k ) : NEW_LINE INDENT k -= 1 NEW_LINE ans . pop ( ) NEW_LINE DEDENT
return ans NEW_LINE
str = " zyxedcba " NEW_LINE k = 1 NEW_LINE print ( * largestString ( str , k ) , sep = " " ) NEW_LINE
def maxLengthSubArray ( A , N ) : NEW_LINE
forward = [ 0 ] * N NEW_LINE backward = [ 0 ] * N NEW_LINE
for i in range ( N ) : NEW_LINE INDENT if i == 0 or A [ i ] != A [ i - 1 ] : NEW_LINE INDENT forward [ i ] = 1 NEW_LINE DEDENT else : NEW_LINE INDENT forward [ i ] = forward [ i - 1 ] + 1 NEW_LINE DEDENT DEDENT
for i in range ( N - 1 , - 1 , - 1 ) : NEW_LINE INDENT if i == N - 1 or A [ i ] != A [ i + 1 ] : NEW_LINE INDENT backward [ i ] = 1 NEW_LINE DEDENT else : NEW_LINE INDENT backward [ i ] = backward [ i + 1 ] + 1 NEW_LINE DEDENT DEDENT
ans = 0 NEW_LINE
for i in range ( N - 1 ) : NEW_LINE INDENT if ( A [ i ] != A [ i + 1 ] ) : NEW_LINE INDENT ans = max ( ans , min ( forward [ i ] , backward [ i + 1 ] ) * 2 ) ; NEW_LINE DEDENT DEDENT
print ( ans ) NEW_LINE
arr = [ 1 , 2 , 3 , 4 , 4 , 4 , 6 , 6 , 6 , 9 ] NEW_LINE
N = len ( arr ) NEW_LINE
maxLengthSubArray ( arr , N ) NEW_LINE
from math import * NEW_LINE
def minNum ( n ) : NEW_LINE INDENT if n < 3 : NEW_LINE INDENT print ( - 1 ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( 210 * ( 10 ** ( n - 1 ) // 210 + 1 ) ) NEW_LINE DEDENT DEDENT
n = 5 NEW_LINE minNum ( n ) NEW_LINE
def helper ( d , s ) : NEW_LINE
ans = [ '0' ] * d NEW_LINE for i in range ( d - 1 , - 1 , - 1 ) : NEW_LINE
if ( s >= 9 ) : NEW_LINE INDENT ans [ i ] = '9' NEW_LINE s -= 9 NEW_LINE DEDENT
else : NEW_LINE INDENT c = chr ( s + ord ( '0' ) ) NEW_LINE ans [ i ] = c ; NEW_LINE s = 0 ; NEW_LINE DEDENT return ' ' . join ( ans ) ; NEW_LINE
def findMin ( x , Y ) : NEW_LINE
y = str ( Y ) ; NEW_LINE n = len ( y ) NEW_LINE p = [ 0 ] * n NEW_LINE
for i in range ( n ) : NEW_LINE INDENT p [ i ] = ( ord ( y [ i ] ) - ord ( '0' ) ) NEW_LINE if ( i > 0 ) : NEW_LINE INDENT p [ i ] += p [ i - 1 ] ; NEW_LINE DEDENT DEDENT
n - 1 NEW_LINE k = 0 NEW_LINE while True : NEW_LINE
d = 0 ; NEW_LINE if ( i >= 0 ) : NEW_LINE INDENT d = ( ord ( y [ i ] ) - ord ( '0' ) ) NEW_LINE DEDENT
for j in range ( d + 1 , 10 ) : NEW_LINE
r = ( ( i > 0 ) * p [ i - 1 ] + j ) ; NEW_LINE
if ( x - r >= 0 and x - r <= 9 * k ) : NEW_LINE
suf = helper ( k , x - r ) ; NEW_LINE pre = " " ; NEW_LINE if ( i > 0 ) : NEW_LINE INDENT pre = y [ 0 : i ] NEW_LINE DEDENT
cur = chr ( j + ord ( '0' ) ) NEW_LINE pre += cur ; NEW_LINE
return pre + suf ; NEW_LINE i -= 1 NEW_LINE k += 1 NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
x = 18 ; NEW_LINE y = 99 ; NEW_LINE
print ( findMin ( x , y ) ) NEW_LINE
def largestNumber ( n , X , Y ) : NEW_LINE INDENT maxm = max ( X , Y ) NEW_LINE DEDENT
Y = X + Y - maxm NEW_LINE
X = maxm NEW_LINE
Xs = 0 NEW_LINE Ys = 0 NEW_LINE while ( n > 0 ) : NEW_LINE
if ( n % Y == 0 ) : NEW_LINE
Xs += n NEW_LINE
n = 0 NEW_LINE else : NEW_LINE
n -= X NEW_LINE
Ys += X NEW_LINE
if ( n == 0 ) : NEW_LINE INDENT while ( Xs > 0 ) : NEW_LINE INDENT Xs -= 1 NEW_LINE print ( X , end = ' ' ) NEW_LINE DEDENT while ( Ys > 0 ) : NEW_LINE INDENT Ys -= 1 NEW_LINE print ( Y , end = ' ' ) NEW_LINE DEDENT DEDENT
else : NEW_LINE INDENT print ( " - 1" ) NEW_LINE DEDENT
n = 19 NEW_LINE X = 7 NEW_LINE Y = 5 NEW_LINE largestNumber ( n , X , Y ) NEW_LINE
def minChanges ( str , N ) : NEW_LINE INDENT count0 = 0 NEW_LINE count1 = 0 NEW_LINE DEDENT
for x in str : NEW_LINE INDENT count0 += ( x == '0' ) NEW_LINE DEDENT res = count0 NEW_LINE
for x in str : NEW_LINE INDENT count0 -= ( x == '0' ) NEW_LINE count1 += ( x == '1' ) NEW_LINE res = min ( res , count1 + count0 ) NEW_LINE DEDENT return res NEW_LINE
N = 9 NEW_LINE str = "000101001" NEW_LINE print ( minChanges ( str , N ) ) NEW_LINE
import sys NEW_LINE
def missingnumber ( n , arr ) -> int : NEW_LINE INDENT mn = sys . maxsize ; NEW_LINE mx = - sys . maxsize - 1 ; NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT if ( i > 0 and arr [ i ] == - 1 and arr [ i - 1 ] != - 1 ) : NEW_LINE INDENT mn = min ( mn , arr [ i - 1 ] ) ; NEW_LINE mx = max ( mx , arr [ i - 1 ] ) ; NEW_LINE DEDENT if ( i < ( n - 1 ) and arr [ i ] == - 1 and arr [ i + 1 ] != - 1 ) : NEW_LINE INDENT mn = min ( mn , arr [ i + 1 ] ) ; NEW_LINE mx = max ( mx , arr [ i + 1 ] ) ; NEW_LINE DEDENT DEDENT res = ( mx + mn ) / 2 ; NEW_LINE return res ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 5 ; NEW_LINE arr = [ - 1 , 10 , - 1 , 12 , - 1 ] ; NEW_LINE DEDENT
res = missingnumber ( n , arr ) ; NEW_LINE print ( res ) ; NEW_LINE
def LCSubStr ( A , B , m , n ) : NEW_LINE
LCSuff = [ [ 0 for i in range ( n + 1 ) ] for j in range ( m + 1 ) ] NEW_LINE result = 0 NEW_LINE
for i in range ( m + 1 ) : NEW_LINE INDENT for j in range ( n + 1 ) : NEW_LINE DEDENT
if ( i == 0 or j == 0 ) : NEW_LINE INDENT LCSuff [ i ] [ j ] = 0 NEW_LINE DEDENT
elif ( A [ i - 1 ] == B [ j - 1 ] ) : NEW_LINE INDENT LCSuff [ i ] [ j ] = LCSuff [ i - 1 ] [ j - 1 ] + 1 NEW_LINE result = max ( result , LCSuff [ i ] [ j ] ) NEW_LINE DEDENT
else : NEW_LINE INDENT LCSuff [ i ] [ j ] = 0 NEW_LINE DEDENT
return result NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT A = "0110" NEW_LINE B = "1101" NEW_LINE M = len ( A ) NEW_LINE N = len ( B ) NEW_LINE DEDENT
print ( LCSubStr ( A , B , M , N ) ) NEW_LINE
maxN = 20 ; NEW_LINE maxSum = 50 ; NEW_LINE minSum = 50 ; NEW_LINE Base = 50 ; NEW_LINE
dp = [ [ 0 for i in range ( maxSum + minSum ) ] for j in range ( maxN ) ] ; NEW_LINE v = [ [ False for i in range ( maxSum + minSum ) ] for j in range ( maxN ) ] ; NEW_LINE
def findCnt ( arr , i , required_sum , n ) : NEW_LINE
if ( i == n ) : NEW_LINE INDENT if ( required_sum == 0 ) : NEW_LINE INDENT return 1 ; NEW_LINE DEDENT else : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT DEDENT
if ( v [ i ] [ required_sum + Base ] ) : NEW_LINE INDENT return dp [ i ] [ required_sum + Base ] ; NEW_LINE DEDENT
v [ i ] [ required_sum + Base ] = True ; NEW_LINE
dp [ i ] [ required_sum + Base ] = findCnt ( arr , i + 1 , required_sum , n ) + findCnt ( arr , i + 1 , required_sum - arr [ i ] , n ) ; NEW_LINE return dp [ i ] [ required_sum + Base ] ; NEW_LINE
def countSubsets ( arr , K , n ) : NEW_LINE
sum = 0 ; NEW_LINE
for i in range ( n ) : NEW_LINE
sum += arr [ i ] ; NEW_LINE
S1 = ( sum + K ) // 2 ; NEW_LINE
print ( findCnt ( arr , 0 , S1 , n ) ) ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 1 , 1 , 2 , 3 ] ; NEW_LINE N = len ( arr ) ; NEW_LINE K = 1 ; NEW_LINE DEDENT
countSubsets ( arr , K , N ) ; NEW_LINE
dp = [ [ 0 for i in range ( 605 ) ] for j in range ( 105 ) ] ; NEW_LINE
def find ( N , sum ) : NEW_LINE INDENT if ( N < 0 sum < 0 ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT if ( dp [ N ] [ sum ] > 0 ) : NEW_LINE INDENT return dp [ N ] [ sum ] ; NEW_LINE DEDENT DEDENT
if ( sum > 6 * N or sum < N ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT if ( N == 1 ) : NEW_LINE INDENT if ( sum >= 1 and sum <= 6 ) : NEW_LINE INDENT return ( float ) ( 1.0 / 6 ) ; NEW_LINE DEDENT else : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT DEDENT for i in range ( 1 , 7 ) : NEW_LINE INDENT dp [ N ] [ sum ] = dp [ N ] [ sum ] + find ( N - 1 , sum - i ) / 6 ; NEW_LINE DEDENT return dp [ N ] [ sum ] ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 4 ; a = 13 ; b = 17 ; NEW_LINE probability = 0.0 NEW_LINE f = 0 ; NEW_LINE DEDENT
for sum in range ( a , b + 1 ) : NEW_LINE INDENT probability = probability + find ( N , sum ) ; NEW_LINE DEDENT
print ( " % .6f " % probability ) ; NEW_LINE
def count ( n ) : NEW_LINE
dp = dict ( ) NEW_LINE
dp [ 0 ] = 0 NEW_LINE dp [ 1 ] = 1 NEW_LINE
if n not in dp : NEW_LINE INDENT dp [ n ] = 1 + min ( n % 2 + count ( n // 2 ) , n % 3 + count ( n // 3 ) ) NEW_LINE DEDENT
return dp [ n ] NEW_LINE
N = 6 NEW_LINE
print ( str ( count ( N ) ) ) NEW_LINE
def find_minimum_operations ( n , b , k ) : NEW_LINE
d = [ 0 for i in range ( n + 1 ) ] NEW_LINE
operations = 0 NEW_LINE for i in range ( n ) : NEW_LINE
d [ i ] += d [ i - 1 ] NEW_LINE
if b [ i ] > d [ i ] : NEW_LINE
operations += ( b [ i ] - d [ i ] ) NEW_LINE need = ( b [ i ] - d [ i ] ) NEW_LINE
d [ i ] += need NEW_LINE
if i + k <= n : NEW_LINE INDENT d [ i + k ] -= need NEW_LINE DEDENT return operations NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 5 NEW_LINE b = [ 1 , 2 , 3 , 4 , 5 ] NEW_LINE k = 2 NEW_LINE DEDENT
print ( find_minimum_operations ( n , b , k ) ) NEW_LINE
def ways ( arr , k ) : NEW_LINE INDENT R = len ( arr ) NEW_LINE C = len ( arr [ 0 ] ) NEW_LINE K = k NEW_LINE preSum = [ [ 0 for _ in range ( C ) ] \ for _ in range ( R ) ] NEW_LINE DEDENT
for r in range ( R - 1 , - 1 , - 1 ) : NEW_LINE INDENT for c in range ( C - 1 , - 1 , - 1 ) : NEW_LINE INDENT preSum [ r ] = arr [ r ] NEW_LINE if r + 1 < R : NEW_LINE INDENT preSum [ r ] += preSum [ r + 1 ] NEW_LINE DEDENT if c + 1 < C : NEW_LINE INDENT preSum [ r ] += preSum [ r ] NEW_LINE DEDENT if r + 1 < R and c + 1 < C : NEW_LINE INDENT preSum [ r ] -= preSum [ r + 1 ] NEW_LINE DEDENT DEDENT DEDENT
dp = [ [ [ 0 for _ in range ( C ) ] \ for _ in range ( R ) ] \ for _ in range ( K + 1 ) ] NEW_LINE
for k in range ( 1 , K + 1 ) : NEW_LINE INDENT for r in range ( R - 1 , - 1 , - 1 ) : NEW_LINE INDENT for c in range ( C - 1 , - 1 , - 1 ) : NEW_LINE INDENT if k == 1 : NEW_LINE INDENT dp [ k ] [ r ] = 1 if preSum [ r ] > 0 else 0 NEW_LINE DEDENT else : NEW_LINE INDENT dp [ k ] [ r ] = 0 NEW_LINE for r1 in range ( r + 1 , R ) : NEW_LINE DEDENT DEDENT DEDENT DEDENT
if preSum [ r ] - preSum [ r1 ] > 0 : NEW_LINE INDENT dp [ k ] [ r ] += dp [ k - 1 ] [ r1 ] NEW_LINE DEDENT for c1 in range ( c + 1 , C ) : NEW_LINE
if preSum [ r ] - preSum [ r ] [ c1 ] > 0 : NEW_LINE INDENT dp [ k ] [ r ] += dp [ k - 1 ] [ r ] [ c1 ] NEW_LINE DEDENT return dp [ K ] [ 0 ] [ 0 ] NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ [ 1 , 0 , 0 ] , [ 1 , 1 , 1 ] , [ 0 , 0 , 0 ] ] NEW_LINE k = 3 NEW_LINE DEDENT
print ( ways ( arr , k ) ) NEW_LINE
p = 1000000007 NEW_LINE
def power ( x , y , p ) : NEW_LINE INDENT res = 1 NEW_LINE x = x % p NEW_LINE while ( y > 0 ) : NEW_LINE DEDENT
if ( y & 1 ) : NEW_LINE INDENT res = ( res * x ) % p NEW_LINE DEDENT
y = y >> 1 NEW_LINE x = ( x * x ) % p NEW_LINE return res NEW_LINE
def nCr ( n , p , f , m ) : NEW_LINE INDENT for i in range ( n + 1 ) : NEW_LINE INDENT for j in range ( m + 1 ) : NEW_LINE DEDENT DEDENT
if ( j > i ) : NEW_LINE INDENT f [ i ] [ j ] = 0 NEW_LINE DEDENT
elif ( j == 0 or j == i ) : NEW_LINE INDENT f [ i ] [ j ] = 1 NEW_LINE DEDENT else : NEW_LINE INDENT f [ i ] [ j ] = ( f [ i - 1 ] [ j ] + f [ i - 1 ] [ j - 1 ] ) % p NEW_LINE DEDENT
def ProductOfSubsets ( arr , n , m ) : NEW_LINE INDENT f = [ [ 0 for i in range ( 100 ) ] for j in range ( n + 1 ) ] NEW_LINE nCr ( n , p - 1 , f , m ) NEW_LINE arr . sort ( reverse = False ) NEW_LINE DEDENT
ans = 1 NEW_LINE for i in range ( n ) : NEW_LINE
x = 0 NEW_LINE for j in range ( 1 , m + 1 , 1 ) : NEW_LINE
if ( m % j == 0 ) : NEW_LINE
x = ( ( x + ( f [ n - i - 1 ] [ m - j ] * f [ i ] [ j - 1 ] ) % ( p - 1 ) ) % ( p - 1 ) ) NEW_LINE ans = ( ( ans * power ( arr [ i ] , x , p ) ) % p ) NEW_LINE print ( ans ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 4 , 5 , 7 , 9 , 3 ] NEW_LINE K = 4 NEW_LINE N = len ( arr ) ; NEW_LINE ProductOfSubsets ( arr , N , K ) NEW_LINE DEDENT
def countWays ( n , m ) : NEW_LINE
dp = [ [ 0 for i in range ( n + 1 ) ] for i in range ( m + 1 ) ] NEW_LINE
for i in range ( n + 1 ) : NEW_LINE INDENT dp [ 1 ] [ i ] = 1 NEW_LINE DEDENT
sum = 0 NEW_LINE for i in range ( 2 , m + 1 ) : NEW_LINE INDENT for j in range ( n + 1 ) : NEW_LINE INDENT sum = 0 NEW_LINE DEDENT DEDENT
for k in range ( j + 1 ) : NEW_LINE INDENT sum += dp [ i - 1 ] [ k ] NEW_LINE DEDENT
dp [ i ] [ j ] = sum NEW_LINE
return dp [ m ] [ n ] NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 2 NEW_LINE K = 3 NEW_LINE DEDENT
print ( countWays ( N , K ) ) NEW_LINE
def countWays ( n , m ) : NEW_LINE
dp = [ [ 0 for i in range ( n + 1 ) ] for j in range ( m + 1 ) ] NEW_LINE
for i in range ( n + 1 ) : NEW_LINE INDENT dp [ 1 ] [ i ] = 1 NEW_LINE if ( i != 0 ) : NEW_LINE INDENT dp [ 1 ] [ i ] += dp [ 1 ] [ i - 1 ] NEW_LINE DEDENT DEDENT
for i in range ( 2 , m + 1 ) : NEW_LINE INDENT for j in range ( n + 1 ) : NEW_LINE DEDENT
if ( j == 0 ) : NEW_LINE INDENT dp [ i ] [ j ] = dp [ i - 1 ] [ j ] NEW_LINE DEDENT
else : NEW_LINE INDENT dp [ i ] [ j ] = dp [ i - 1 ] [ j ] NEW_LINE DEDENT
if ( i == m and j == n ) : NEW_LINE INDENT return dp [ i ] [ j ] NEW_LINE DEDENT
dp [ i ] [ j ] += dp [ i ] [ j - 1 ] NEW_LINE
N = 2 NEW_LINE K = 3 NEW_LINE
print ( countWays ( N , K ) ) NEW_LINE
from math import sqrt NEW_LINE
def SieveOfEratosthenes ( MAX , primes ) : NEW_LINE INDENT prime = [ True ] * ( MAX + 1 ) ; NEW_LINE DEDENT
for p in range ( 2 , int ( sqrt ( MAX ) ) + 1 ) : NEW_LINE INDENT if ( prime [ p ] == True ) : NEW_LINE DEDENT
for i in range ( p ** 2 , MAX + 1 , p ) : NEW_LINE INDENT prime [ i ] = False ; NEW_LINE DEDENT
for i in range ( 2 , MAX + 1 ) : NEW_LINE INDENT if ( prime [ i ] ) : NEW_LINE INDENT primes . append ( i ) ; NEW_LINE DEDENT DEDENT
def findLongest ( A , n ) : NEW_LINE
mpp = { } ; NEW_LINE primes = [ ] ; NEW_LINE
SieveOfEratosthenes ( A [ n - 1 ] , primes ) ; NEW_LINE dp = [ 0 ] * n ; NEW_LINE
dp [ n - 1 ] = 1 ; NEW_LINE mpp [ A [ n - 1 ] ] = n - 1 ; NEW_LINE
for i in range ( n - 2 , - 1 , - 1 ) : NEW_LINE
num = A [ i ] ; NEW_LINE
dp [ i ] = 1 ; NEW_LINE maxi = 0 ; NEW_LINE
for it in primes : NEW_LINE
xx = num * it ; NEW_LINE
if ( xx > A [ n - 1 ] ) : NEW_LINE INDENT break ; NEW_LINE DEDENT
elif xx in mpp : NEW_LINE
dp [ i ] = max ( dp [ i ] , 1 + dp [ mpp [ xx ] ] ) ; NEW_LINE
mpp [ A [ i ] ] = i ; NEW_LINE ans = 1 ; NEW_LINE
for i in range ( n ) : NEW_LINE INDENT ans = max ( ans , dp [ i ] ) ; NEW_LINE DEDENT return ans ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = [ 1 , 2 , 5 , 6 , 12 , 35 , 60 , 385 ] ; NEW_LINE n = len ( a ) ; NEW_LINE print ( findLongest ( a , n ) ) ; NEW_LINE DEDENT
def waysToKAdjacentSetBits ( n , k , currentIndex , adjacentSetBits , lastBit ) : NEW_LINE
if ( currentIndex == n ) : NEW_LINE
if ( adjacentSetBits == k ) : NEW_LINE INDENT return 1 ; NEW_LINE DEDENT return 0 NEW_LINE noOfWays = 0 NEW_LINE
if ( lastBit == 1 ) : NEW_LINE
noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits + 1 , 1 ) ; NEW_LINE
noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits , 0 ) ; NEW_LINE elif ( lastBit != 1 ) : NEW_LINE noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits , 1 ) ; NEW_LINE noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits , 0 ) ; NEW_LINE return noOfWays ; NEW_LINE
n = 5 ; k = 2 ; NEW_LINE
totalWays = ( waysToKAdjacentSetBits ( n , k , 1 , 0 , 1 ) + waysToKAdjacentSetBits ( n , k , 1 , 0 , 0 ) ) ; NEW_LINE print ( " Number ▁ of ▁ ways ▁ = " , totalWays ) ; NEW_LINE
def postfix ( a , n ) : NEW_LINE INDENT for i in range ( n - 1 , 1 , - 1 ) : NEW_LINE INDENT a [ i - 1 ] = a [ i - 1 ] + a [ i ] NEW_LINE DEDENT DEDENT
def modify ( a , n ) : NEW_LINE INDENT for i in range ( 1 , n ) : NEW_LINE INDENT a [ i - 1 ] = i * a [ i ] ; NEW_LINE DEDENT DEDENT
def allCombination ( a , n ) : NEW_LINE INDENT sum = 0 NEW_LINE DEDENT
for i in range ( 1 , n + 1 ) : NEW_LINE INDENT sum += i NEW_LINE DEDENT print ( " f ( 1 ) ▁ - - > ▁ " , sum ) NEW_LINE
for i in range ( 1 , n ) : NEW_LINE
postfix ( a , n - i + 1 ) NEW_LINE
sum = 0 NEW_LINE for j in range ( 1 , n - i + 1 ) : NEW_LINE INDENT sum += ( j * a [ j ] ) NEW_LINE DEDENT print ( " f ( " , i + 1 , " ) ▁ - - > ▁ " , sum ) NEW_LINE
modify ( a , n ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 5 NEW_LINE a = [ 0 ] * n NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT a [ i ] = i + 1 NEW_LINE DEDENT
allCombination ( a , n ) NEW_LINE
def findStep ( n ) : NEW_LINE INDENT if ( n == 1 or n == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT elif ( n == 2 ) : NEW_LINE INDENT return 2 NEW_LINE DEDENT else : NEW_LINE INDENT return findStep ( n - 3 ) + findStep ( n - 2 ) + findStep ( n - 1 ) NEW_LINE DEDENT DEDENT
n = 4 NEW_LINE print ( findStep ( n ) ) NEW_LINE
def isSubsetSum ( arr , n , sum ) : NEW_LINE
if sum == 0 : NEW_LINE INDENT return True NEW_LINE DEDENT if n == 0 and sum != 0 : NEW_LINE INDENT return False NEW_LINE DEDENT
if arr [ n - 1 ] > sum : NEW_LINE INDENT return isSubsetSum ( arr , n - 1 , sum ) NEW_LINE DEDENT
return isSubsetSum ( arr , n - 1 , sum ) or isSubsetSum ( arr , n - 1 , sum - arr [ n - 1 ] ) NEW_LINE
def findPartion ( arr , n ) : NEW_LINE
sum = 0 NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT sum += arr [ i ] NEW_LINE DEDENT
if sum % 2 != 0 : NEW_LINE INDENT return false NEW_LINE DEDENT
return isSubsetSum ( arr , n , sum // 2 ) NEW_LINE
arr = [ 3 , 1 , 5 , 9 , 12 ] NEW_LINE n = len ( arr ) NEW_LINE
if findPartion ( arr , n ) == True : NEW_LINE INDENT print ( " Can ▁ be ▁ divided ▁ into ▁ two ▁ subsets ▁ of ▁ equal ▁ sum " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Can ▁ not ▁ be ▁ divided ▁ into ▁ two ▁ subsets ▁ of ▁ equal ▁ sum " ) NEW_LINE DEDENT
def findPartiion ( arr , n ) : NEW_LINE INDENT Sum = 0 NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT Sum += arr [ i ] NEW_LINE DEDENT if ( Sum % 2 != 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT part = [ 0 ] * ( ( Sum // 2 ) + 1 ) NEW_LINE
for i in range ( ( Sum // 2 ) + 1 ) : NEW_LINE INDENT part [ i ] = 0 NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE
for j in range ( Sum // 2 , arr [ i ] - 1 , - 1 ) : NEW_LINE
if ( part [ j - arr [ i ] ] == 1 or j == arr [ i ] ) : NEW_LINE INDENT part [ j ] = 1 NEW_LINE DEDENT return part [ Sum // 2 ] NEW_LINE
arr = [ 1 , 3 , 3 , 2 , 3 , 2 ] NEW_LINE n = len ( arr ) NEW_LINE
if ( findPartiion ( arr , n ) == 1 ) : NEW_LINE INDENT print ( " Can ▁ be ▁ divided ▁ into ▁ two ▁ subsets ▁ of ▁ equal ▁ sum " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Can ▁ not ▁ be ▁ divided ▁ into ▁ two ▁ subsets ▁ of ▁ equal ▁ sum " ) NEW_LINE DEDENT
def binomialCoeff ( n , r ) : NEW_LINE INDENT if ( r > n ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT m = 1000000007 NEW_LINE inv = [ 0 for i in range ( r + 1 ) ] NEW_LINE inv [ 0 ] = 1 ; NEW_LINE if ( r + 1 >= 2 ) NEW_LINE inv [ 1 ] = 1 ; NEW_LINE DEDENT
for i in range ( 2 , r + 1 ) : NEW_LINE INDENT inv [ i ] = m - ( m // i ) * inv [ m % i ] % m NEW_LINE DEDENT ans = 1 NEW_LINE
for i in range ( 2 , r + 1 ) : NEW_LINE INDENT ans = ( ( ans % m ) * ( inv [ i ] % m ) ) % m NEW_LINE DEDENT
for i in range ( n , n - r , - 1 ) : NEW_LINE INDENT ans = ( ( ans % m ) * ( i % m ) ) % m NEW_LINE DEDENT return ans NEW_LINE
n = 5 NEW_LINE r = 2 NEW_LINE print ( " Value ▁ of ▁ C ( " , n , " , ▁ " , r , " ) ▁ is ▁ " , binomialCoeff ( n , r ) ) NEW_LINE
def gcd ( a , b ) : NEW_LINE
if ( a < b ) : NEW_LINE INDENT t = a NEW_LINE a = b NEW_LINE b = t NEW_LINE DEDENT if ( a % b == 0 ) : NEW_LINE INDENT return b NEW_LINE DEDENT
return gcd ( b , a % b ) NEW_LINE
def printAnswer ( x , y ) : NEW_LINE
val = gcd ( x , y ) NEW_LINE
if ( ( val & ( val - 1 ) ) == 0 ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE
x = 4 NEW_LINE y = 7 NEW_LINE
printAnswer ( x , y ) NEW_LINE
def getElement ( N , r , c ) : NEW_LINE
if ( r > c ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT
if ( r == 1 ) : NEW_LINE INDENT return c ; NEW_LINE DEDENT
a = ( r + 1 ) * pow ( 2 , r - 2 ) ; NEW_LINE
d = pow ( 2 , r - 1 ) ; NEW_LINE
c = c - r ; NEW_LINE element = a + d * c ; NEW_LINE return element ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT N = 4 ; R = 3 ; C = 4 ; NEW_LINE print ( getElement ( N , R , C ) ) ; NEW_LINE DEDENT
def MinValue ( N , X ) : NEW_LINE
N = list ( N ) ; NEW_LINE ln = len ( N ) NEW_LINE
position = ln + 1 NEW_LINE
if ( N [ 0 ] == ' - ' ) : NEW_LINE
for i in range ( ln - 1 , 0 , - 1 ) : NEW_LINE INDENT if ( ( ord ( N [ i ] ) - ord ( '0' ) ) < X ) : NEW_LINE INDENT position = i NEW_LINE DEDENT DEDENT else : NEW_LINE
for i in range ( ln - 1 , - 1 , - 1 ) : NEW_LINE INDENT if ( ( ord ( N [ i ] ) - ord ( '0' ) ) > X ) : NEW_LINE INDENT position = i NEW_LINE DEDENT DEDENT
c = chr ( X + ord ( '0' ) ) NEW_LINE str = N . insert ( position , c ) ; NEW_LINE
return ' ' . join ( N ) NEW_LINE
N = "89" NEW_LINE X = 1 NEW_LINE
print ( MinValue ( N , X ) ) NEW_LINE
def divisibleByk ( s , n , k ) : NEW_LINE
poweroftwo = [ 0 for i in range ( n ) ] NEW_LINE
poweroftwo [ 0 ] = 1 % k NEW_LINE for i in range ( 1 , n , 1 ) : NEW_LINE
poweroftwo [ i ] = ( poweroftwo [ i - 1 ] * ( 2 % k ) ) % k NEW_LINE
rem = 0 NEW_LINE
for i in range ( n ) : NEW_LINE
if ( s [ n - i - 1 ] == '1' ) : NEW_LINE
rem += ( poweroftwo [ i ] ) NEW_LINE rem %= k NEW_LINE
if ( rem == 0 ) : NEW_LINE INDENT return " Yes " NEW_LINE DEDENT
else : NEW_LINE INDENT return " No " NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE
s = "1010001" NEW_LINE k = 9 NEW_LINE
n = len ( s ) NEW_LINE
print ( divisibleByk ( s , n , k ) ) NEW_LINE
def maxSumbySplittingstring ( str , N ) : NEW_LINE
cntOne = 0 NEW_LINE
for i in range ( N ) : NEW_LINE
' NEW_LINE INDENT if ( str [ i ] == '1' ) : NEW_LINE DEDENT
cntOne += 1 NEW_LINE
zero = 0 NEW_LINE
one = 0 NEW_LINE
res = 0 NEW_LINE
for i in range ( N - 1 ) : NEW_LINE
' NEW_LINE INDENT if ( str [ i ] == '0' ) : NEW_LINE DEDENT
zero += 1 NEW_LINE
' NEW_LINE INDENT else : NEW_LINE DEDENT
one += 1 NEW_LINE
res = max ( res , zero + cntOne - one ) NEW_LINE return res NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT str = "00111" NEW_LINE N = len ( str ) NEW_LINE print ( maxSumbySplittingstring ( str , N ) ) NEW_LINE DEDENT
def cntBalancedParenthesis ( s , N ) : NEW_LINE
cntPairs = 0 ; NEW_LINE
cntCurly = 0 ; NEW_LINE
cntSml = 0 ; NEW_LINE
cntSqr = 0 ; NEW_LINE
for i in range ( N ) : NEW_LINE INDENT if ( ord ( s [ i ] ) == ord ( ' { ' ) ) : NEW_LINE DEDENT
cntCurly += 1 ; NEW_LINE elif ( ord ( s [ i ] ) == ord ( ' ( ' ) ) : NEW_LINE
cntSml += 1 ; NEW_LINE elif ( ord ( s [ i ] ) == ord ( ' [ ' ) ) : NEW_LINE
cntSqr += 1 ; NEW_LINE elif ( ord ( s [ i ] ) == ord ( ' } ' ) and cntCurly > 0 ) : NEW_LINE
cntCurly -= 1 ; NEW_LINE
cntPairs += 1 ; NEW_LINE elif ( ord ( s [ i ] ) == ord ( ' ) ' ) and cntSml > 0 ) : NEW_LINE
cntSml -= 1 ; NEW_LINE
cntPairs += 1 ; NEW_LINE elif ( ord ( s [ i ] ) == ord ( ' ] ' ) and cntSqr > 0 ) : NEW_LINE
cntSqr -= 1 ; NEW_LINE
cntPairs += 1 ; NEW_LINE print ( cntPairs ) ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
s = " { ( } ) " ; NEW_LINE N = len ( s ) ; NEW_LINE
cntBalancedParenthesis ( s , N ) ; NEW_LINE
def arcIntersection ( S , lenn ) : NEW_LINE INDENT stk = [ ] NEW_LINE DEDENT
for i in range ( lenn ) : NEW_LINE
stk . append ( S [ i ] ) NEW_LINE if ( len ( stk ) >= 2 ) : NEW_LINE
temp = stk [ - 1 ] NEW_LINE
del stk [ - 1 ] NEW_LINE
if ( stk [ - 1 ] == temp ) : NEW_LINE INDENT del stk [ - 1 ] NEW_LINE DEDENT
else : NEW_LINE INDENT stk . append ( temp ) NEW_LINE DEDENT
if ( len ( stk ) == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT return 0 NEW_LINE
def countString ( arr , N ) : NEW_LINE
count = 0 NEW_LINE
for i in range ( N ) : NEW_LINE
lenn = len ( arr [ i ] ) NEW_LINE
count += arcIntersection ( arr [ i ] , lenn ) NEW_LINE
print ( count ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ "0101" , "0011" , "0110" ] NEW_LINE N = len ( arr ) NEW_LINE DEDENT
countString ( arr , N ) NEW_LINE
def ConvertequivalentBase8 ( S ) : NEW_LINE
mp = { } NEW_LINE
mp [ "000" ] = '0' NEW_LINE mp [ "001" ] = '1' NEW_LINE mp [ "010" ] = '2' NEW_LINE mp [ "011" ] = '3' NEW_LINE mp [ "100" ] = '4' NEW_LINE mp [ "101" ] = '5' NEW_LINE mp [ "110" ] = '6' NEW_LINE mp [ "111" ] = '7' NEW_LINE
N = len ( S ) NEW_LINE if ( N % 3 == 2 ) : NEW_LINE
S = "0" + S NEW_LINE elif ( N % 3 == 1 ) : NEW_LINE
S = "00" + S NEW_LINE
N = len ( S ) NEW_LINE
octal = " " NEW_LINE
for i in range ( 0 , N , 3 ) : NEW_LINE
temp = S [ i : i + 3 ] NEW_LINE
if temp in mp : NEW_LINE octal += ( mp [ temp ] ) NEW_LINE return octal NEW_LINE
def binString_div_9 ( S , N ) : NEW_LINE
octal = ConvertequivalentBase8 ( S ) NEW_LINE
oddSum = 0 NEW_LINE
evenSum = 0 NEW_LINE
M = len ( octal ) NEW_LINE
for i in range ( 0 , M , 2 ) : NEW_LINE
oddSum += ord ( octal [ i ] ) - ord ( '0' ) NEW_LINE
for i in range ( 1 , M , 2 ) : NEW_LINE
evenSum += ord ( octal [ i ] ) - ord ( '0' ) NEW_LINE
Oct_9 = 11 NEW_LINE
if ( abs ( oddSum - evenSum ) % Oct_9 == 0 ) : NEW_LINE INDENT return " Yes " NEW_LINE DEDENT return " No " NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT S = "1010001" NEW_LINE N = len ( S ) NEW_LINE print ( binString_div_9 ( S , N ) ) NEW_LINE DEDENT
def min_cost ( S ) : NEW_LINE
cost = 0 NEW_LINE
F = 0 NEW_LINE
B = 0 NEW_LINE
n = len ( S ) - S . count ( ' ▁ ' ) NEW_LINE
if n == 1 : NEW_LINE INDENT return cost NEW_LINE DEDENT
for char in S : NEW_LINE
if char != ' ▁ ' : NEW_LINE
if B != 0 : NEW_LINE
cost += min ( n - F , F ) * B NEW_LINE B = 0 NEW_LINE
F += 1 NEW_LINE
else : NEW_LINE
B += 1 NEW_LINE
return cost NEW_LINE
S = " ▁ @ TABSYMBOL $ " NEW_LINE print ( min_cost ( S ) ) NEW_LINE
def isVowel ( ch ) : NEW_LINE INDENT if ( ch == ' a ' or ch == ' e ' or ch == ' i ' or ch == ' o ' or ch == ' u ' ) : NEW_LINE INDENT return True ; NEW_LINE DEDENT else : NEW_LINE INDENT return False ; NEW_LINE DEDENT DEDENT
def minCost ( S ) : NEW_LINE
cA = 0 ; NEW_LINE cE = 0 ; NEW_LINE cI = 0 ; NEW_LINE cO = 0 ; NEW_LINE cU = 0 ; NEW_LINE
for i in range ( len ( S ) ) : NEW_LINE
if ( isVowel ( S [ i ] ) ) : NEW_LINE
cA += abs ( ord ( S [ i ] ) - ord ( ' a ' ) ) ; NEW_LINE cE += abs ( ord ( S [ i ] ) - ord ( ' e ' ) ) ; NEW_LINE cI += abs ( ord ( S [ i ] ) - ord ( ' i ' ) ) ; NEW_LINE cO += abs ( ord ( S [ i ] ) - ord ( ' o ' ) ) ; NEW_LINE cU += abs ( ord ( S [ i ] ) - ord ( ' u ' ) ) ; NEW_LINE
return min ( min ( min ( min ( cA , cE ) , cI ) , cO ) , cU ) ; NEW_LINE
S = " geeksforgeeks " ; NEW_LINE print ( minCost ( S ) ) NEW_LINE
def decode_String ( st , K ) : NEW_LINE INDENT ans = " " NEW_LINE DEDENT
for i in range ( 0 , len ( st ) , K ) : NEW_LINE
ans += st [ i ] NEW_LINE
for i in range ( len ( st ) - ( K - 1 ) , len ( st ) ) : NEW_LINE INDENT ans += st [ i ] NEW_LINE DEDENT print ( ans ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT K = 3 NEW_LINE st = " abcbcscsesesesd " NEW_LINE decode_String ( st , K ) NEW_LINE DEDENT
def maxVowelSubString ( str1 , K ) : NEW_LINE
N = len ( str1 ) NEW_LINE
pref = [ 0 for i in range ( N ) ] NEW_LINE
for i in range ( N ) : NEW_LINE
if ( str1 [ i ] == ' a ' or str1 [ i ] == ' e ' or str1 [ i ] == ' i ' or str1 [ i ] == ' o ' or str1 [ i ] == ' u ' ) : NEW_LINE INDENT pref [ i ] = 1 NEW_LINE DEDENT
else : NEW_LINE INDENT pref [ i ] = 0 NEW_LINE DEDENT
if ( i ) : NEW_LINE INDENT pref [ i ] += pref [ i - 1 ] NEW_LINE DEDENT
maxCount = pref [ K - 1 ] NEW_LINE
res = str1 [ 0 : K ] NEW_LINE
for i in range ( K , N ) : NEW_LINE
currCount = pref [ i ] - pref [ i - K ] NEW_LINE
if ( currCount > maxCount ) : NEW_LINE INDENT maxCount = currCount NEW_LINE res = str1 [ i - K + 1 : i + 1 ] NEW_LINE DEDENT
elif ( currCount == maxCount ) : NEW_LINE INDENT temp = str1 [ i - K + 1 : i + 1 ] NEW_LINE if ( temp < res ) : NEW_LINE INDENT res = temp NEW_LINE DEDENT DEDENT
return res NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT str1 = " ceebbaceeffo " NEW_LINE K = 3 NEW_LINE print ( maxVowelSubString ( str1 , K ) ) NEW_LINE DEDENT
def decodeStr ( str , len ) : NEW_LINE
c = [ " " for i in range ( len ) ] NEW_LINE pos = 1 NEW_LINE
if ( len % 2 == 1 ) : NEW_LINE INDENT med = int ( len / 2 ) NEW_LINE DEDENT else : NEW_LINE INDENT med = int ( len / 2 - 1 ) NEW_LINE DEDENT
c [ med ] = str [ 0 ] NEW_LINE
if ( len % 2 == 0 ) : NEW_LINE INDENT c [ med + 1 ] = str [ 1 ] NEW_LINE DEDENT
if ( len & 1 ) : NEW_LINE INDENT k = 1 NEW_LINE DEDENT else : NEW_LINE INDENT k = 2 NEW_LINE DEDENT for i in range ( k , len , 2 ) : NEW_LINE INDENT c [ med - pos ] = str [ i ] NEW_LINE DEDENT
if ( len % 2 == 1 ) : NEW_LINE INDENT c [ med + pos ] = str [ i + 1 ] NEW_LINE DEDENT
else : NEW_LINE INDENT c [ med + pos + 1 ] = str [ i + 1 ] NEW_LINE DEDENT pos += 1 NEW_LINE
print ( * c , sep = " " ) NEW_LINE
str = " ofrsgkeeeekgs " NEW_LINE len = len ( str ) NEW_LINE decodeStr ( str , len ) NEW_LINE
def findCount ( s , L , R ) : NEW_LINE
distinct = 0 NEW_LINE
frequency = [ 0 for i in range ( 26 ) ] NEW_LINE
for i in range ( L , R + 1 , 1 ) : NEW_LINE
frequency [ ord ( s [ i ] ) - ord ( ' a ' ) ] += 1 NEW_LINE for i in range ( 26 ) : NEW_LINE
if ( frequency [ i ] > 0 ) : NEW_LINE INDENT distinct += 1 NEW_LINE DEDENT print ( distinct ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT s = " geeksforgeeksisacomputerscienceportal " NEW_LINE queries = 3 NEW_LINE Q = [ [ 0 , 10 ] , [ 15 , 18 ] , [ 12 , 20 ] ] NEW_LINE for i in range ( queries ) : NEW_LINE INDENT findCount ( s , Q [ i ] [ 0 ] , Q [ i ] [ 1 ] ) NEW_LINE DEDENT DEDENT
def ReverseComplement ( s , n , k ) : NEW_LINE
rev = ( k + 1 ) // 2 NEW_LINE
complement = k - rev NEW_LINE
if ( rev % 2 ) : NEW_LINE INDENT s = s [ : : - 1 ] NEW_LINE DEDENT
if ( complement % 2 ) : NEW_LINE INDENT for i in range ( n ) : NEW_LINE DEDENT
if ( s [ i ] == '0' ) : NEW_LINE INDENT s [ i ] = '1' NEW_LINE DEDENT else : NEW_LINE INDENT s [ i ] = '0' NEW_LINE DEDENT
return s NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT str1 = "10011" NEW_LINE k = 5 NEW_LINE n = len ( str1 ) NEW_LINE DEDENT
print ( ReverseComplement ( str1 , n , k ) ) NEW_LINE
def repeatingString ( s , n , k ) : NEW_LINE
if ( n % k != 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT
frequency = [ 0 for i in range ( 123 ) ] NEW_LINE
for i in range ( 123 ) : NEW_LINE INDENT frequency [ i ] = 0 NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT frequency [ s [ i ] ] += 1 NEW_LINE DEDENT repeat = n // k NEW_LINE
for i in range ( 123 ) : NEW_LINE INDENT if ( frequency [ i ] % repeat != 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT return True NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT s = " abcdcba " NEW_LINE n = len ( s ) NEW_LINE k = 3 NEW_LINE if ( repeatingString ( s , n , k ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT DEDENT
def findPhoneNumber ( n ) : NEW_LINE INDENT temp = n NEW_LINE sum = 0 NEW_LINE DEDENT
while ( temp != 0 ) : NEW_LINE INDENT sum += temp % 10 NEW_LINE temp = temp // 10 NEW_LINE DEDENT
if ( sum < 10 ) : NEW_LINE INDENT print ( n , "0" , sum ) NEW_LINE DEDENT
else : NEW_LINE INDENT n = str ( n ) NEW_LINE sum = str ( sum ) NEW_LINE n += sum NEW_LINE print ( n ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 98765432 NEW_LINE findPhoneNumber ( n ) NEW_LINE DEDENT
def cntSplits ( s ) : NEW_LINE
if ( s [ len ( s ) - 1 ] == '1' ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT
c_zero = 0 ; NEW_LINE
for i in range ( len ( s ) ) : NEW_LINE INDENT c_zero += ( s [ i ] == '0' ) ; NEW_LINE DEDENT
return int ( pow ( 2 , c_zero - 1 ) ) ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT s = "10010" ; NEW_LINE print ( cntSplits ( s ) ) ; NEW_LINE DEDENT
def findNumbers ( s ) : NEW_LINE
n = len ( s ) NEW_LINE
count = 1 NEW_LINE result = 0 NEW_LINE
left = 0 NEW_LINE right = 1 NEW_LINE while ( right < n ) : NEW_LINE
if ( s [ left ] == s [ right ] ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT
else : NEW_LINE
result += count * ( count + 1 ) // 2 NEW_LINE
left = right NEW_LINE count = 1 NEW_LINE right += 1 NEW_LINE
result += count * ( count + 1 ) // 2 NEW_LINE print ( result ) NEW_LINE
s = " bbbcbb " NEW_LINE findNumbers ( s ) NEW_LINE
def isVowel ( ch ) : NEW_LINE INDENT ch = ch . upper ( ) NEW_LINE if ( ch == ' A ' or ch == ' E ' or ch == ' I ' or ch == ' O ' or ch == ' U ' ) : NEW_LINE INDENT return True NEW_LINE DEDENT else : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT
def duplicateVowels ( S ) : NEW_LINE INDENT t = len ( S ) NEW_LINE DEDENT
res = " " NEW_LINE
for i in range ( t ) : NEW_LINE INDENT if ( isVowel ( S [ i ] ) ) : NEW_LINE INDENT res += S [ i ] NEW_LINE DEDENT res += S [ i ] NEW_LINE DEDENT return res NEW_LINE
S = " helloworld " NEW_LINE
print ( " Original ▁ String : ▁ " , S ) NEW_LINE res = duplicateVowels ( S ) NEW_LINE
print ( " String ▁ with ▁ Vowels ▁ duplicated : ▁ " , res ) NEW_LINE
def stringToInt ( str ) : NEW_LINE
if ( len ( str ) == 1 ) : NEW_LINE INDENT return ord ( str [ 0 ] ) - ord ( '0' ) ; NEW_LINE DEDENT
y = stringToInt ( str [ 1 : ] ) ; NEW_LINE
x = ord ( str [ 0 ] ) - ord ( '0' ) ; NEW_LINE
x = x * ( 10 ** ( len ( str ) - 1 ) ) + y ; NEW_LINE return int ( x ) ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT str = "1235" ; NEW_LINE print ( stringToInt ( str ) ) ; NEW_LINE DEDENT
MAX = 26 NEW_LINE
def largestSubSeq ( arr , n ) : NEW_LINE
count = [ 0 ] * MAX NEW_LINE
for i in range ( n ) : NEW_LINE INDENT string = arr [ i ] NEW_LINE DEDENT
_hash = [ False ] * MAX NEW_LINE for j in range ( len ( string ) ) : NEW_LINE INDENT _hash [ ord ( string [ j ] ) - ord ( ' a ' ) ] = True NEW_LINE DEDENT for j in range ( MAX ) : NEW_LINE
if _hash [ j ] == True : NEW_LINE INDENT count [ j ] += 1 NEW_LINE DEDENT return max ( count ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ " ab " , " bc " , " de " ] NEW_LINE n = len ( arr ) NEW_LINE print ( largestSubSeq ( arr , n ) ) NEW_LINE DEDENT
def isPalindrome ( s ) : NEW_LINE INDENT l = len ( s ) NEW_LINE for i in range ( l // 2 ) : NEW_LINE INDENT if ( s [ i ] != s [ l - 1 - i ] ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT return True NEW_LINE DEDENT
def createStringAndCheckPalindrome ( N ) : NEW_LINE
sub = " " + chr ( N ) NEW_LINE res_str = " " NEW_LINE sum = 0 NEW_LINE
while ( N > 0 ) : NEW_LINE INDENT digit = N % 10 NEW_LINE sum += digit NEW_LINE N = N // 10 NEW_LINE DEDENT
while ( len ( res_str ) < sum ) : NEW_LINE INDENT res_str += sub NEW_LINE DEDENT
if ( len ( res_str ) > sum ) : NEW_LINE INDENT res_str = res_str [ 0 : sum ] NEW_LINE DEDENT
if ( isPalindrome ( res_str ) ) : NEW_LINE INDENT return True NEW_LINE DEDENT return False NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT N = 10101 NEW_LINE if ( createStringAndCheckPalindrome ( N ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT DEDENT
def minimumLength ( s ) : NEW_LINE INDENT maxOcc = 0 NEW_LINE n = len ( s ) NEW_LINE arr = [ 0 ] * 26 NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT arr [ ord ( s [ i ] ) - ord ( ' a ' ) ] += 1 NEW_LINE DEDENT
for i in range ( 26 ) : NEW_LINE INDENT if arr [ i ] > maxOcc : NEW_LINE INDENT maxOcc = arr [ i ] NEW_LINE DEDENT DEDENT
return n - maxOcc NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT str = " afddewqd " NEW_LINE print ( minimumLength ( str ) ) NEW_LINE DEDENT
def removeSpecialCharacter ( s ) : NEW_LINE INDENT i = 0 NEW_LINE while i < len ( s ) : NEW_LINE DEDENT
if ( ord ( s [ i ] ) < ord ( ' A ' ) or ord ( s [ i ] ) > ord ( ' Z ' ) and ord ( s [ i ] ) < ord ( ' a ' ) or ord ( s [ i ] ) > ord ( ' z ' ) ) : NEW_LINE
del s [ i ] NEW_LINE i -= 1 NEW_LINE i += 1 NEW_LINE print ( " " . join ( s ) ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT s = " $ Gee * k ; s . . fo , ▁ r ' Ge ^ eks ? " NEW_LINE s = [ i for i in s ] NEW_LINE removeSpecialCharacter ( s ) NEW_LINE DEDENT
def removeSpecialCharacter ( s ) : NEW_LINE INDENT t = " " NEW_LINE for i in s : NEW_LINE DEDENT
if ( i >= ' A ' and i <= ' Z ' ) or ( i >= ' a ' and i <= ' z ' ) : NEW_LINE INDENT t += i NEW_LINE DEDENT print ( t ) NEW_LINE
s = " $ Gee * k ; s . . fo , ▁ r ' Ge ^ eks ? " NEW_LINE removeSpecialCharacter ( s ) NEW_LINE
def findRepeatFirstN2 ( s ) : NEW_LINE
p = - 1 NEW_LINE for i in range ( len ( s ) ) : NEW_LINE INDENT for j in range ( i + 1 , len ( s ) ) : NEW_LINE INDENT if ( s [ i ] == s [ j ] ) : NEW_LINE INDENT p = i NEW_LINE break NEW_LINE DEDENT DEDENT if ( p != - 1 ) : NEW_LINE INDENT break NEW_LINE DEDENT DEDENT return p NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT str = " geeksforgeeks " NEW_LINE pos = findRepeatFirstN2 ( str ) NEW_LINE if ( pos == - 1 ) : NEW_LINE INDENT print ( " Not ▁ found " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( str [ pos ] ) NEW_LINE DEDENT DEDENT
def prCharWithFreq ( str ) : NEW_LINE
d = { } NEW_LINE for i in str : NEW_LINE INDENT if i in d : NEW_LINE INDENT d [ i ] += 1 NEW_LINE DEDENT else : NEW_LINE INDENT d [ i ] = 1 NEW_LINE DEDENT DEDENT
for i in str : NEW_LINE
if d [ i ] != 0 : NEW_LINE INDENT print ( " { } { } " . format ( i , d [ i ] ) , end = " ▁ " ) NEW_LINE d [ i ] = 0 NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT str = " geeksforgeeks " ; NEW_LINE prCharWithFreq ( str ) ; NEW_LINE DEDENT ' NEW_LINE
def possibleStrings ( n , r , b , g ) : NEW_LINE
fact = [ 0 for i in range ( n + 1 ) ] NEW_LINE fact [ 0 ] = 1 NEW_LINE for i in range ( 1 , n + 1 , 1 ) : NEW_LINE INDENT fact [ i ] = fact [ i - 1 ] * i NEW_LINE DEDENT
left = n - ( r + g + b ) NEW_LINE sum = 0 NEW_LINE
for i in range ( 0 , left + 1 , 1 ) : NEW_LINE INDENT for j in range ( 0 , left - i + 1 , 1 ) : NEW_LINE INDENT k = left - ( i + j ) NEW_LINE DEDENT DEDENT
sum = ( sum + fact [ n ] / ( fact [ i + r ] * fact [ j + b ] * fact [ k + g ] ) ) NEW_LINE
return sum NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 4 NEW_LINE r = 2 NEW_LINE b = 0 NEW_LINE g = 1 NEW_LINE print ( int ( possibleStrings ( n , r , b , g ) ) ) NEW_LINE DEDENT
CHARS = 26 NEW_LINE
def remAnagram ( str1 , str2 ) : NEW_LINE
count1 = [ 0 ] * CHARS NEW_LINE count2 = [ 0 ] * CHARS NEW_LINE
i = 0 NEW_LINE while i < len ( str1 ) : NEW_LINE INDENT count1 [ ord ( str1 [ i ] ) - ord ( ' a ' ) ] += 1 NEW_LINE i += 1 NEW_LINE DEDENT
i = 0 NEW_LINE while i < len ( str2 ) : NEW_LINE INDENT count2 [ ord ( str2 [ i ] ) - ord ( ' a ' ) ] += 1 NEW_LINE i += 1 NEW_LINE DEDENT
result = 0 NEW_LINE for i in range ( 26 ) : NEW_LINE INDENT result += abs ( count1 [ i ] - count2 [ i ] ) NEW_LINE DEDENT return result NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT str1 = " bcadeh " NEW_LINE str2 = " hea " NEW_LINE print ( remAnagram ( str1 , str2 ) ) NEW_LINE DEDENT
CHARS = 26 NEW_LINE
def isValidString ( str ) : NEW_LINE INDENT freq = [ 0 ] * CHARS NEW_LINE DEDENT
for i in range ( len ( str ) ) : NEW_LINE INDENT freq [ ord ( str [ i ] ) - ord ( ' a ' ) ] += 1 NEW_LINE DEDENT
freq1 = 0 NEW_LINE count_freq1 = 0 NEW_LINE for i in range ( CHARS ) : NEW_LINE INDENT if ( freq [ i ] != 0 ) : NEW_LINE INDENT freq1 = freq [ i ] NEW_LINE count_freq1 = 1 NEW_LINE break NEW_LINE DEDENT DEDENT
freq2 = 0 NEW_LINE count_freq2 = 0 NEW_LINE for j in range ( i + 1 , CHARS ) : NEW_LINE INDENT if ( freq [ j ] != 0 ) : NEW_LINE INDENT if ( freq [ j ] == freq1 ) : NEW_LINE INDENT count_freq1 += 1 NEW_LINE DEDENT else : NEW_LINE INDENT count_freq2 = 1 NEW_LINE freq2 = freq [ j ] NEW_LINE break NEW_LINE DEDENT DEDENT DEDENT
for k in range ( j + 1 , CHARS ) : NEW_LINE INDENT if ( freq [ k ] != 0 ) : NEW_LINE INDENT if ( freq [ k ] == freq1 ) : NEW_LINE INDENT count_freq1 += 1 NEW_LINE DEDENT if ( freq [ k ] == freq2 ) : NEW_LINE INDENT count_freq2 += 1 NEW_LINE DEDENT DEDENT DEDENT
else : NEW_LINE INDENT return False NEW_LINE DEDENT
if ( count_freq1 > 1 and count_freq2 > 1 ) : NEW_LINE INDENT return False NEW_LINE DEDENT
return True NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT str = " abcbc " NEW_LINE if ( isValidString ( str ) ) : NEW_LINE INDENT print ( " YES " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NO " ) NEW_LINE DEDENT DEDENT
def checkForVariation ( strr ) : NEW_LINE INDENT if ( len ( strr ) == 0 ) : NEW_LINE INDENT return True NEW_LINE DEDENT mapp = { } NEW_LINE DEDENT
for i in range ( len ( strr ) ) : NEW_LINE INDENT if strr [ i ] in mapp : NEW_LINE INDENT mapp [ strr [ i ] ] += 1 NEW_LINE DEDENT else : NEW_LINE INDENT mapp [ strr [ i ] ] = 1 NEW_LINE DEDENT DEDENT
first = True NEW_LINE second = True NEW_LINE val1 = 0 NEW_LINE val2 = 0 NEW_LINE countOfVal1 = 0 NEW_LINE countOfVal2 = 0 NEW_LINE for itr in mapp : NEW_LINE INDENT i = itr NEW_LINE DEDENT
if ( first ) : NEW_LINE INDENT val1 = i NEW_LINE first = False NEW_LINE countOfVal1 += 1 NEW_LINE continue NEW_LINE DEDENT if ( i == val1 ) : NEW_LINE INDENT countOfVal1 += 1 NEW_LINE continue NEW_LINE DEDENT
if ( second ) : NEW_LINE INDENT val2 = i NEW_LINE countOfVal2 += 1 NEW_LINE second = False NEW_LINE continue NEW_LINE DEDENT if ( i == val2 ) : NEW_LINE INDENT countOfVal2 += 1 NEW_LINE continue NEW_LINE DEDENT if ( countOfVal1 > 1 and countOfVal2 > 1 ) : NEW_LINE return False NEW_LINE else : NEW_LINE return True NEW_LINE
print ( checkForVariation ( " abcbc " ) ) NEW_LINE
def countCompletePairs ( set1 , set2 , n , m ) : NEW_LINE INDENT result = 0 NEW_LINE DEDENT
con_s1 , con_s2 = [ 0 ] * n , [ 0 ] * m NEW_LINE
for i in range ( n ) : NEW_LINE
con_s1 [ i ] = 0 NEW_LINE for j in range ( len ( set1 [ i ] ) ) : NEW_LINE
con_s1 [ i ] = con_s1 [ i ] | ( 1 << ( ord ( set1 [ i ] [ j ] ) - ord ( ' a ' ) ) ) NEW_LINE
for i in range ( m ) : NEW_LINE
con_s2 [ i ] = 0 NEW_LINE for j in range ( len ( set2 [ i ] ) ) : NEW_LINE
con_s2 [ i ] = con_s2 [ i ] | ( 1 << ( ord ( set2 [ i ] [ j ] ) - ord ( ' a ' ) ) ) NEW_LINE
complete = ( 1 << 26 ) - 1 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT for j in range ( m ) : NEW_LINE DEDENT
if ( ( con_s1 [ i ] con_s2 [ j ] ) == complete ) : NEW_LINE INDENT result += 1 NEW_LINE DEDENT return result NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT set1 = [ " abcdefgh " , " geeksforgeeks " , " lmnopqrst " , " abc " ] NEW_LINE set2 = [ " ijklmnopqrstuvwxyz " , " abcdefghijklmnopqrstuvwxyz " , " defghijklmnopqrstuvwxyz " ] NEW_LINE n = len ( set1 ) NEW_LINE m = len ( set2 ) NEW_LINE print ( countCompletePairs ( set1 , set2 , n , m ) ) NEW_LINE DEDENT
def encodeString ( Str ) : NEW_LINE INDENT map = { } NEW_LINE res = " " NEW_LINE i = 0 NEW_LINE DEDENT
for ch in Str : NEW_LINE
if ch not in map : NEW_LINE INDENT map [ ch ] = i NEW_LINE i += 1 NEW_LINE DEDENT
res += str ( map [ ch ] ) NEW_LINE return res NEW_LINE
def findMatchedWords ( dict , pattern ) : NEW_LINE
Len = len ( pattern ) NEW_LINE
hash = encodeString ( pattern ) NEW_LINE
for word in dict : NEW_LINE
if ( len ( word ) == Len and encodeString ( word ) == hash ) : NEW_LINE INDENT print ( word , end = " ▁ " ) NEW_LINE DEDENT
dict = [ " abb " , " abc " , " xyz " , " xyy " ] NEW_LINE pattern = " foo " NEW_LINE findMatchedWords ( dict , pattern ) NEW_LINE
def check ( pattern , word ) : NEW_LINE INDENT if ( len ( pattern ) != len ( word ) ) : NEW_LINE INDENT return False NEW_LINE DEDENT ch = [ 0 for i in range ( 128 ) ] NEW_LINE Len = len ( word ) NEW_LINE for i in range ( Len ) : NEW_LINE INDENT if ( ch [ ord ( pattern [ i ] ) ] == 0 ) : NEW_LINE INDENT ch [ ord ( pattern [ i ] ) ] = word [ i ] NEW_LINE DEDENT elif ( ch [ ord ( pattern [ i ] ) ] != word [ i ] ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT return True NEW_LINE DEDENT
def findMatchedWords ( Dict , pattern ) : NEW_LINE
Len = len ( pattern ) NEW_LINE
for word in range ( len ( Dict ) - 1 , - 1 , - 1 ) : NEW_LINE INDENT if ( check ( pattern , Dict [ word ] ) ) : NEW_LINE INDENT print ( Dict [ word ] , end = " ▁ " ) NEW_LINE DEDENT DEDENT
Dict = [ " abb " , " abc " , " xyz " , " xyy " ] NEW_LINE pattern = " foo " NEW_LINE findMatchedWords ( Dict , pattern ) NEW_LINE
def countWords ( Str ) : NEW_LINE
if ( Str == None or len ( Str ) == 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT wordCount = 0 NEW_LINE isWord = False NEW_LINE endOfLine = len ( Str ) - 1 NEW_LINE
ch = list ( Str ) NEW_LINE for i in range ( len ( ch ) ) : NEW_LINE
if ( ch [ i ] . isalpha ( ) and i != endOfLine ) : NEW_LINE INDENT isWord = True NEW_LINE DEDENT
elif ( not ch [ i ] . isalpha ( ) and isWord ) : NEW_LINE INDENT wordCount += 1 NEW_LINE isWord = False NEW_LINE DEDENT
elif ( ch [ i ] . isalpha ( ) and i == endOfLine ) : NEW_LINE INDENT wordCount += 1 NEW_LINE DEDENT
return wordCount NEW_LINE
Str =   " One two three NEW_LINE INDENT four five   " NEW_LINE DEDENT
print ( " No ▁ of ▁ words ▁ : " , countWords ( Str ) ) NEW_LINE
def RevString ( s , l ) : NEW_LINE
INDENT if l % 2 == 0 : NEW_LINE DEDENT
j = int ( l / 2 ) NEW_LINE
while ( j <= l - 1 ) : NEW_LINE s [ j ] , s [ l - j - 1 ] = s [ l - j - 1 ] , s [ j ] NEW_LINE j += 1 NEW_LINE
INDENT else : NEW_LINE DEDENT
j = int ( l / 2 + 1 ) NEW_LINE
while ( j <= l - 1 ) : NEW_LINE s [ j ] , s [ l - 1 - j ] = s [ l - j - 1 ] , s [ j ] NEW_LINE j += 1 NEW_LINE
return s ; NEW_LINE
s = ' getting ▁ good ▁ at ▁ coding ▁ needs ▁ a ▁ lot ▁ of ▁ practice ' NEW_LINE string = s . split ( ' ▁ ' ) NEW_LINE string = RevString ( string , len ( string ) ) NEW_LINE print ( " ▁ " . join ( string ) ) NEW_LINE
def printPath ( res , nThNode , kThNode ) : NEW_LINE
if kThNode > nThNode : NEW_LINE INDENT return NEW_LINE DEDENT
res . append ( kThNode ) NEW_LINE
for i in range ( 0 , len ( res ) ) : NEW_LINE INDENT print ( res [ i ] , end = " ▁ " ) NEW_LINE DEDENT print ( ) NEW_LINE
printPath ( res [ : ] , nThNode , kThNode * 2 ) NEW_LINE
printPath ( res [ : ] , nThNode , kThNode * 2 + 1 ) NEW_LINE
def printPathToCoverAllNodeUtil ( nThNode ) : NEW_LINE
res = [ ] NEW_LINE
printPath ( res , nThNode , 1 ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
nThNode = 7 NEW_LINE
printPathToCoverAllNodeUtil ( nThNode ) NEW_LINE
import math NEW_LINE
def getMid ( s : int , e : int ) -> int : NEW_LINE INDENT return s + ( e - s ) // 2 NEW_LINE DEDENT
def isArmstrong ( x : int ) -> bool : NEW_LINE INDENT n = len ( str ( x ) ) NEW_LINE sum1 = 0 NEW_LINE temp = x NEW_LINE while ( temp > 0 ) : NEW_LINE INDENT digit = temp % 10 NEW_LINE sum1 += pow ( digit , n ) NEW_LINE temp //= 10 NEW_LINE DEDENT if ( sum1 == x ) : NEW_LINE INDENT return True NEW_LINE DEDENT return False NEW_LINE DEDENT
def MaxUtil ( st , ss , se , l , r , node ) : NEW_LINE
if ( l <= ss and r >= se ) : NEW_LINE INDENT return st [ node ] NEW_LINE DEDENT
if ( se < l or ss > r ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
mid = getMid ( ss , se ) NEW_LINE return max ( MaxUtil ( st , ss , mid , l , r , 2 * node + 1 ) , MaxUtil ( st , mid + 1 , se , l , r , 2 * node + 2 ) ) NEW_LINE
def updateValue ( arr , st , ss , se , index , value , node ) : NEW_LINE INDENT if ( index < ss or index > se ) : NEW_LINE INDENT print ( " Invalid ▁ Input " ) NEW_LINE return NEW_LINE DEDENT if ( ss == se ) : NEW_LINE DEDENT
arr [ index ] = value NEW_LINE if ( isArmstrong ( value ) ) : NEW_LINE INDENT st [ node ] = value NEW_LINE DEDENT else : NEW_LINE INDENT st [ node ] = - 1 NEW_LINE DEDENT else : NEW_LINE mid = getMid ( ss , se ) NEW_LINE if ( index >= ss and index <= mid ) : NEW_LINE INDENT updateValue ( arr , st , ss , mid , index , value , 2 * node + 1 ) NEW_LINE DEDENT else : NEW_LINE INDENT updateValue ( arr , st , mid + 1 , se , index , value , 2 * node + 2 ) NEW_LINE DEDENT st [ node ] = max ( st [ 2 * node + 1 ] , st [ 2 * node + 2 ] ) NEW_LINE return NEW_LINE
def getMax ( st , n , l , r ) : NEW_LINE
if ( l < 0 or r > n - 1 or l > r ) : NEW_LINE INDENT print ( " Invalid ▁ Input " ) NEW_LINE return - 1 NEW_LINE DEDENT return MaxUtil ( st , 0 , n - 1 , l , r , 0 ) NEW_LINE
def constructSTUtil ( arr , ss , se , st , si ) : NEW_LINE
if ( ss == se ) : NEW_LINE INDENT if ( isArmstrong ( arr [ ss ] ) ) : NEW_LINE INDENT st [ si ] = arr [ ss ] NEW_LINE DEDENT else : NEW_LINE INDENT st [ si ] = - 1 NEW_LINE DEDENT return st [ si ] NEW_LINE DEDENT
mid = getMid ( ss , se ) NEW_LINE st [ si ] = max ( constructSTUtil ( arr , ss , mid , st , si * 2 + 1 ) , constructSTUtil ( arr , mid + 1 , se , st , si * 2 + 2 ) ) NEW_LINE return st [ si ] NEW_LINE
def constructST ( arr , n ) : NEW_LINE
x = int ( math . ceil ( math . log2 ( n ) ) ) NEW_LINE
max_size = 2 * int ( math . pow ( 2 , x ) ) - 1 NEW_LINE
st = [ 0 for _ in range ( max_size ) ] NEW_LINE
constructSTUtil ( arr , 0 , n - 1 , st , 0 ) NEW_LINE
return st NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 192 , 113 , 535 , 7 , 19 , 111 ] NEW_LINE n = len ( arr ) NEW_LINE DEDENT
st = constructST ( arr , n ) NEW_LINE
print ( " Maximum ▁ armstrong ▁ number ▁ in ▁ given ▁ range ▁ = ▁ { } " . format ( getMax ( st , n , 1 , 3 ) ) ) NEW_LINE
updateValue ( arr , st , 0 , n - 1 , 1 , 153 , 0 ) NEW_LINE
print ( " Updated ▁ Maximum ▁ armstrong ▁ number ▁ in ▁ given ▁ range ▁ = ▁ { } " . format ( getMax ( st , n , 1 , 3 ) ) ) NEW_LINE
def maxRegions ( n ) : NEW_LINE INDENT num = n * ( n + 1 ) // 2 + 1 NEW_LINE DEDENT
print ( num ) NEW_LINE
n = 10 NEW_LINE maxRegions ( n ) NEW_LINE
def checkSolveable ( n , m ) : NEW_LINE
if n == 1 or m == 1 : NEW_LINE INDENT print ( " YES " ) NEW_LINE DEDENT
elif m == 2 and n == 2 : NEW_LINE INDENT print ( " YES " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NO " ) NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 1 NEW_LINE m = 3 NEW_LINE checkSolveable ( n , m ) NEW_LINE DEDENT
def GCD ( a , b ) : NEW_LINE
if ( b == 0 ) : NEW_LINE INDENT return a NEW_LINE DEDENT
else : NEW_LINE INDENT return GCD ( b , a % b ) NEW_LINE DEDENT
def check ( x , y ) : NEW_LINE
if ( GCD ( x , y ) == 1 ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE
X = 2 NEW_LINE Y = 7 NEW_LINE
check ( X , Y ) NEW_LINE
size = 1000001 NEW_LINE
def seiveOfEratosthenes ( prime ) : NEW_LINE INDENT prime [ 0 ] = 1 NEW_LINE prime [ 1 ] = 0 NEW_LINE i = 2 NEW_LINE while ( i * i < 1000001 ) : NEW_LINE DEDENT
if ( prime [ i ] == 0 ) : NEW_LINE INDENT j = i * i NEW_LINE while ( j < 1000001 ) : NEW_LINE DEDENT
prime [ j ] = 1 NEW_LINE j = j + i NEW_LINE i += 1 NEW_LINE
def probabiltyEuler ( prime , L , R , M ) : NEW_LINE INDENT arr = [ 0 ] * size NEW_LINE eulerTotient = [ 0 ] * size NEW_LINE count = 0 NEW_LINE DEDENT
for i in range ( L , R + 1 ) : NEW_LINE
eulerTotient [ i - L ] = i NEW_LINE arr [ i - L ] = i NEW_LINE for i in range ( 2 , 1000001 ) : NEW_LINE
if ( prime [ i ] == 0 ) : NEW_LINE
for j in range ( ( L // i ) * i , R + 1 , i ) : NEW_LINE INDENT if ( j - L >= 0 ) : NEW_LINE DEDENT
eulerTotient [ j - L ] = ( eulerTotient [ j - L ] // i * ( i - 1 ) ) NEW_LINE while ( arr [ j - L ] % i == 0 ) : NEW_LINE INDENT arr [ j - L ] = arr [ j - L ] // i NEW_LINE DEDENT
for i in range ( L , R + 1 ) : NEW_LINE INDENT if ( arr [ i - L ] > 1 ) : NEW_LINE INDENT eulerTotient [ i - L ] = ( ( eulerTotient [ i - L ] // arr [ i - L ] ) * ( arr [ i - L ] - 1 ) ) NEW_LINE DEDENT DEDENT for i in range ( L , R + 1 ) : NEW_LINE
if ( ( eulerTotient [ i - L ] % M ) == 0 ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT
return ( float ) ( 1.0 * count / ( R + 1 - L ) ) NEW_LINE
prime = [ 0 ] * size NEW_LINE seiveOfEratosthenes ( prime ) NEW_LINE L , R , M = 1 , 7 , 3 NEW_LINE print ( probabiltyEuler ( prime , L , R , M ) ) NEW_LINE
import math NEW_LINE
def findWinner ( n , k ) : NEW_LINE INDENT cnt = 0 ; NEW_LINE DEDENT
if ( n == 1 ) : NEW_LINE INDENT print ( " No " ) ; NEW_LINE DEDENT
elif ( ( n & 1 ) or n == 2 ) : NEW_LINE INDENT print ( " Yes " ) ; NEW_LINE DEDENT else : NEW_LINE INDENT tmp = n ; NEW_LINE val = 1 ; NEW_LINE DEDENT
while ( tmp > k and tmp % 2 == 0 ) : NEW_LINE INDENT tmp //= 2 ; NEW_LINE val *= 2 ; NEW_LINE DEDENT
for i in range ( 3 , int ( math . sqrt ( tmp ) ) + 1 ) : NEW_LINE INDENT while ( tmp % i == 0 ) : NEW_LINE INDENT cnt += 1 ; NEW_LINE tmp //= i ; NEW_LINE DEDENT DEDENT if ( tmp > 1 ) : NEW_LINE INDENT cnt += 1 ; NEW_LINE DEDENT
if ( val == n ) : NEW_LINE INDENT print ( " No " ) ; NEW_LINE DEDENT elif ( n / tmp == 2 and cnt == 1 ) : NEW_LINE INDENT print ( " No " ) ; NEW_LINE DEDENT
else : NEW_LINE INDENT print ( " Yes " ) ; NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 1 ; k = 1 ; NEW_LINE findWinner ( n , k ) ; NEW_LINE DEDENT
import math NEW_LINE
def pen_hex ( n ) : NEW_LINE INDENT pn = 1 NEW_LINE for i in range ( 1 , N ) : NEW_LINE DEDENT
pn = ( int ) ( i * ( 3 * i - 1 ) / 2 ) NEW_LINE if ( pn > n ) : NEW_LINE INDENT break NEW_LINE DEDENT
seqNum = ( 1 + math . sqrt ( 8 * pn + 1 ) ) / 4 NEW_LINE if ( seqNum == ( int ) ( seqNum ) ) : NEW_LINE INDENT print ( pn , end = " , ▁ " ) NEW_LINE DEDENT
N = 1000000 NEW_LINE pen_hex ( N ) NEW_LINE
def isPal ( a , n , m ) : NEW_LINE
for i in range ( 0 , n // 2 ) : NEW_LINE INDENT for j in range ( 0 , m - 1 ) : NEW_LINE INDENT if ( a [ i ] [ j ] != a [ n - 1 - i ] [ m - 1 - j ] ) : NEW_LINE INDENT return False ; NEW_LINE DEDENT DEDENT DEDENT return True ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 3 ; NEW_LINE m = 3 ; NEW_LINE a = [ [ 1 , 2 , 3 ] , [ 4 , 5 , 4 ] , [ 3 , 2 , 1 ] ] ; NEW_LINE if ( isPal ( a , n , m ) ) : NEW_LINE INDENT print ( " YES " ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NO " ) ; NEW_LINE DEDENT DEDENT
def getSum ( n ) : NEW_LINE INDENT sum1 = 0 ; NEW_LINE while ( n != 0 ) : NEW_LINE INDENT sum1 = sum1 + n % 10 ; NEW_LINE n = n // 10 ; NEW_LINE DEDENT return sum1 ; NEW_LINE DEDENT
def smallestNumber ( N ) : NEW_LINE INDENT i = 1 ; NEW_LINE while ( 1 ) : NEW_LINE DEDENT
if ( getSum ( i ) == N ) : NEW_LINE INDENT print ( i ) ; NEW_LINE break ; NEW_LINE DEDENT i += 1 ; NEW_LINE
N = 10 ; NEW_LINE smallestNumber ( N ) ; NEW_LINE
import math NEW_LINE
def reversDigits ( num ) : NEW_LINE INDENT rev_num = 0 NEW_LINE while ( num > 0 ) : NEW_LINE INDENT rev_num = rev_num * 10 + num % 10 NEW_LINE num = num // 10 NEW_LINE DEDENT return rev_num NEW_LINE DEDENT
def isPerfectSquare ( x ) : NEW_LINE
sr = math . sqrt ( x ) NEW_LINE
return ( ( sr - int ( sr ) ) == 0 ) NEW_LINE
def isRare ( N ) : NEW_LINE
reverseN = reversDigits ( N ) NEW_LINE
if ( reverseN == N ) : NEW_LINE INDENT return False NEW_LINE DEDENT return ( isPerfectSquare ( N + reverseN ) and isPerfectSquare ( N - reverseN ) ) NEW_LINE
N = 65 NEW_LINE if ( isRare ( N ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
def calc_ans ( l , r ) : NEW_LINE INDENT power2 = [ ] ; power3 = [ ] ; NEW_LINE DEDENT
mul2 = 1 ; NEW_LINE while ( mul2 <= r ) : NEW_LINE INDENT power2 . append ( mul2 ) ; NEW_LINE mul2 *= 2 ; NEW_LINE DEDENT
mul3 = 1 ; NEW_LINE while ( mul3 <= r ) : NEW_LINE INDENT power3 . append ( mul3 ) ; NEW_LINE mul3 *= 3 ; NEW_LINE DEDENT
power23 = [ ] ; NEW_LINE for x in range ( len ( power2 ) ) : NEW_LINE INDENT for y in range ( len ( power3 ) ) : NEW_LINE INDENT mul = power2 [ x ] * power3 [ y ] ; NEW_LINE if ( mul == 1 ) : NEW_LINE INDENT continue ; NEW_LINE DEDENT DEDENT DEDENT
if ( mul <= r ) : NEW_LINE INDENT power23 . append ( mul ) ; NEW_LINE DEDENT
ans = 0 ; NEW_LINE for x in power23 : NEW_LINE INDENT if ( x >= l and x <= r ) : NEW_LINE INDENT ans += 1 ; NEW_LINE DEDENT DEDENT
print ( ans ) ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT l = 1 ; r = 10 ; NEW_LINE calc_ans ( l , r ) ; NEW_LINE DEDENT
def nCr ( n , r ) : NEW_LINE INDENT if ( r > n ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT return fact ( n ) // ( fact ( r ) * fact ( n - r ) ) NEW_LINE DEDENT
def fact ( n ) : NEW_LINE INDENT res = 1 NEW_LINE for i in range ( 2 , n + 1 ) : NEW_LINE INDENT res = res * i NEW_LINE DEDENT return res NEW_LINE DEDENT
def countSubsequences ( arr , n , k ) : NEW_LINE INDENT countOdd = 0 NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT if ( arr [ i ] & 1 ) : NEW_LINE INDENT countOdd += 1 ; NEW_LINE DEDENT DEDENT ans = nCr ( n , k ) - nCr ( countOdd , k ) ; NEW_LINE return ans NEW_LINE
arr = [ 2 , 4 ] NEW_LINE K = 1 NEW_LINE N = len ( arr ) NEW_LINE print ( countSubsequences ( arr , N , K ) ) NEW_LINE
import math NEW_LINE
def first_digit ( x , y ) : NEW_LINE
length = int ( math . log ( x ) / math . log ( y ) + 1 ) NEW_LINE
first_digit = x / math . pow ( y , length - 1 ) NEW_LINE print ( int ( first_digit ) ) NEW_LINE
X = 55 NEW_LINE Y = 3 NEW_LINE first_digit ( X , Y ) NEW_LINE
def checkIfCurzonNumber ( N ) : NEW_LINE INDENT powerTerm , productTerm = 0 , 0 NEW_LINE DEDENT
powerTerm = pow ( 2 , N ) + 1 NEW_LINE
productTerm = 2 * N + 1 NEW_LINE
if ( powerTerm % productTerm == 0 ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 5 NEW_LINE checkIfCurzonNumber ( N ) NEW_LINE N = 10 NEW_LINE checkIfCurzonNumber ( N ) NEW_LINE DEDENT
def minCount ( n ) : NEW_LINE
hasharr = [ 10 , 3 , 6 , 9 , 2 , 5 , 8 , 1 , 4 , 7 ] NEW_LINE
if ( n > 69 ) : NEW_LINE INDENT return hasharr [ n % 10 ] NEW_LINE DEDENT else : NEW_LINE
if ( n >= hasharr [ n % 10 ] * 7 ) : NEW_LINE INDENT return hasharr [ n % 10 ] NEW_LINE DEDENT else : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
n = 38 ; NEW_LINE print ( minCount ( n ) ) NEW_LINE
def modifiedBinaryPattern ( n ) : NEW_LINE
for i in range ( 1 , n + 1 , 1 ) : NEW_LINE
for j in range ( 1 , i + 1 , 1 ) : NEW_LINE
if ( j == 1 or j == i ) : NEW_LINE INDENT print ( 1 , end = " " ) NEW_LINE DEDENT
else : NEW_LINE INDENT print ( 0 , end = " " ) NEW_LINE DEDENT
print ( ' ' , end = " " ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 7 NEW_LINE DEDENT
modifiedBinaryPattern ( n ) NEW_LINE
def findRealAndImag ( s ) : NEW_LINE
l = len ( s ) NEW_LINE
i = 0 NEW_LINE
' NEW_LINE INDENT if ( s . find ( ' + ' ) != - 1 ) : NEW_LINE INDENT i = s . find ( ' + ' ) NEW_LINE DEDENT DEDENT
' NEW_LINE INDENT else : NEW_LINE INDENT i = s . find ( ' - ' ) ; NEW_LINE DEDENT DEDENT
real = s [ : i ] NEW_LINE
imaginary = s [ i + 1 : l - 1 ] NEW_LINE print ( " Real ▁ part : " , real ) NEW_LINE print ( " Imaginary ▁ part : " , imaginary ) NEW_LINE
s = "3 + 4i " ; NEW_LINE findRealAndImag ( s ) ; NEW_LINE
from math import pow NEW_LINE
def highestPower ( n , k ) : NEW_LINE INDENT i = 0 NEW_LINE a = pow ( n , i ) NEW_LINE DEDENT
while ( a <= k ) : NEW_LINE INDENT i += 1 NEW_LINE a = pow ( n , i ) NEW_LINE DEDENT return i - 1 NEW_LINE
b = [ 0 for i in range ( 50 ) ] NEW_LINE
def PowerArray ( n , k ) : NEW_LINE INDENT while ( k ) : NEW_LINE DEDENT
t = highestPower ( n , k ) NEW_LINE
if ( b [ t ] ) : NEW_LINE
print ( - 1 ) NEW_LINE return 0 NEW_LINE else : NEW_LINE
b [ t ] = 1 NEW_LINE
k -= pow ( n , t ) NEW_LINE
for i in range ( 50 ) : NEW_LINE INDENT if ( b [ i ] ) : NEW_LINE INDENT print ( i , end = ' , ▁ ' ) NEW_LINE DEDENT DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 3 NEW_LINE K = 40 NEW_LINE PowerArray ( N , K ) NEW_LINE DEDENT
N = 100005 NEW_LINE
def SieveOfEratosthenes ( composite ) : NEW_LINE INDENT for p in range ( 2 , N ) : NEW_LINE INDENT if p * p > N : NEW_LINE INDENT break NEW_LINE DEDENT DEDENT DEDENT
if ( composite [ p ] == False ) : NEW_LINE
for i in range ( 2 * p , N , p ) : NEW_LINE INDENT composite [ i ] = True NEW_LINE DEDENT
def sumOfElements ( arr , n ) : NEW_LINE INDENT composite = [ False ] * N NEW_LINE SieveOfEratosthenes ( composite ) NEW_LINE DEDENT
m = dict ( ) ; NEW_LINE for i in range ( n ) : NEW_LINE INDENT m [ arr [ i ] ] = m . get ( arr [ i ] , 0 ) + 1 NEW_LINE DEDENT
sum = 0 NEW_LINE
for it in m : NEW_LINE
if ( composite [ m [ it ] ] ) : NEW_LINE INDENT sum += ( it ) NEW_LINE DEDENT return sum NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 1 , 2 , 1 , 1 , 1 , 3 , 3 , 2 , 4 ] NEW_LINE n = len ( arr ) NEW_LINE DEDENT
print ( sumOfElements ( arr , n ) ) NEW_LINE
def remove ( arr , n ) : NEW_LINE
m = dict . fromkeys ( arr , 0 ) ; NEW_LINE for i in range ( n ) : NEW_LINE INDENT m [ arr [ i ] ] += 1 ; NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE
if ( ( m [ arr [ i ] ] & 1 ) ) : NEW_LINE INDENT continue ; NEW_LINE DEDENT print ( arr [ i ] , end = " , ▁ " ) ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 3 , 3 , 3 , 2 , 2 , 4 , 7 , 7 ] ; NEW_LINE n = len ( arr ) ; NEW_LINE DEDENT
remove ( arr , n ) ; NEW_LINE
def getmax ( arr , n , x ) : NEW_LINE
s = 0 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT s = s + arr [ i ] NEW_LINE DEDENT
print ( min ( s , x ) ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 1 , 2 , 3 , 4 ] NEW_LINE x = 5 NEW_LINE arr_size = len ( arr ) NEW_LINE getmax ( arr , arr_size , x ) NEW_LINE DEDENT
def shortestLength ( n , x , y ) : NEW_LINE INDENT answer = 0 NEW_LINE DEDENT
i = 0 NEW_LINE while n > 0 : NEW_LINE
if ( x [ i ] + y [ i ] > answer ) : NEW_LINE INDENT answer = x [ i ] + y [ i ] NEW_LINE DEDENT i += 1 NEW_LINE n -= 1 NEW_LINE
print ( " Length ▁ - > ▁ " + str ( answer ) ) NEW_LINE print ( " Path ▁ - > ▁ " + " ( ▁ 1 , ▁ " + str ( answer ) + " ▁ ) " + " and ▁ ( ▁ " + str ( answer ) + " , ▁ 1 ▁ ) " ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
n = 4 NEW_LINE
x = [ 1 , 4 , 2 , 1 ] NEW_LINE y = [ 4 , 1 , 1 , 2 ] NEW_LINE shortestLength ( n , x , y ) NEW_LINE
def FindPoints ( x1 , y1 , x2 , y2 , x3 , y3 , x4 , y4 ) : NEW_LINE
x5 = max ( x1 , x3 ) NEW_LINE y5 = max ( y1 , y3 ) NEW_LINE
x6 = min ( x2 , x4 ) NEW_LINE y6 = min ( y2 , y4 ) NEW_LINE
if ( x5 > x6 or y5 > y6 ) : NEW_LINE INDENT print ( " No ▁ intersection " ) NEW_LINE return NEW_LINE DEDENT print ( " ( " , x5 , " , ▁ " , y5 , " ) ▁ " , end = " ▁ " ) NEW_LINE print ( " ( " , x6 , " , ▁ " , y6 , " ) ▁ " , end = " ▁ " ) NEW_LINE
x7 = x5 NEW_LINE y7 = y6 NEW_LINE print ( " ( " , x7 , " , ▁ " , y7 , " ) ▁ " , end = " ▁ " ) NEW_LINE
x8 = x6 NEW_LINE y8 = y5 NEW_LINE print ( " ( " , x8 , " , ▁ " , y8 , " ) ▁ " ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
x1 = 0 NEW_LINE y1 = 0 NEW_LINE x2 = 10 NEW_LINE y2 = 8 NEW_LINE
x3 = 2 NEW_LINE y3 = 3 NEW_LINE x4 = 7 NEW_LINE y4 = 9 NEW_LINE
FindPoints ( x1 , y1 , x2 , y2 , x3 , y3 , x4 , y4 ) NEW_LINE
import math NEW_LINE
class Point : NEW_LINE INDENT def __init__ ( self , a = 0 , b = 0 ) : NEW_LINE INDENT self . x = a NEW_LINE self . y = b NEW_LINE DEDENT DEDENT
def printCorners ( p , q , l ) : NEW_LINE INDENT a , b , c , d = Point ( ) , Point ( ) , Point ( ) , Point ( ) NEW_LINE DEDENT
if ( p . x == q . x ) : NEW_LINE INDENT a . x = p . x - ( l / 2.0 ) NEW_LINE a . y = p . y NEW_LINE d . x = p . x + ( l / 2.0 ) NEW_LINE d . y = p . y NEW_LINE b . x = q . x - ( l / 2.0 ) NEW_LINE b . y = q . y NEW_LINE c . x = q . x + ( l / 2.0 ) NEW_LINE c . y = q . y NEW_LINE DEDENT
elif ( p . y == q . y ) : NEW_LINE INDENT a . y = p . y - ( l / 2.0 ) NEW_LINE a . x = p . x NEW_LINE d . y = p . y + ( l / 2.0 ) NEW_LINE d . x = p . x NEW_LINE b . y = q . y - ( l / 2.0 ) NEW_LINE b . x = q . x NEW_LINE c . y = q . y + ( l / 2.0 ) NEW_LINE c . x = q . x NEW_LINE DEDENT
else : NEW_LINE
m = ( p . x - q . x ) / ( q . y - p . y ) NEW_LINE
dx = ( l / math . sqrt ( 1 + ( m * m ) ) ) * 0.5 NEW_LINE dy = m * dx NEW_LINE a . x = p . x - dx NEW_LINE a . y = p . y - dy NEW_LINE d . x = p . x + dx NEW_LINE d . y = p . y + dy NEW_LINE b . x = q . x - dx NEW_LINE b . y = q . y - dy NEW_LINE c . x = q . x + dx NEW_LINE c . y = q . y + dy NEW_LINE print ( int ( a . x ) , " , ▁ " , int ( a . y ) , sep = " " ) NEW_LINE print ( int ( b . x ) , " , ▁ " , int ( b . y ) , sep = " " ) NEW_LINE print ( int ( c . x ) , " , ▁ " , int ( c . y ) , sep = " " ) NEW_LINE print ( int ( d . x ) , " , ▁ " , int ( d . y ) , sep = " " ) NEW_LINE print ( ) NEW_LINE
p1 = Point ( 1 , 0 ) NEW_LINE q1 = Point ( 1 , 2 ) NEW_LINE printCorners ( p1 , q1 , 2 ) NEW_LINE p = Point ( 1 , 1 ) NEW_LINE q = Point ( - 1 , - 1 ) NEW_LINE printCorners ( p , q , 2 * math . sqrt ( 2 ) ) NEW_LINE
def minimumCost ( arr , N , X , Y ) : NEW_LINE
even_count = 0 NEW_LINE odd_count = 0 NEW_LINE for i in range ( 0 , N ) : NEW_LINE
if ( ( arr [ i ] & 1 ) and ( i % 2 == 0 ) ) : NEW_LINE INDENT odd_count += 1 NEW_LINE DEDENT
if ( ( arr [ i ] % 2 ) == 0 and ( i & 1 ) ) : NEW_LINE INDENT even_count += 1 NEW_LINE DEDENT
cost1 = X * min ( odd_count , even_count ) NEW_LINE
cost2 = Y * ( max ( odd_count , even_count ) - min ( odd_count , even_count ) ) NEW_LINE
cost3 = ( odd_count + even_count ) * Y NEW_LINE
return min ( cost1 + cost2 , cost3 ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 5 , 3 , 7 , 2 , 1 ] NEW_LINE X = 10 NEW_LINE Y = 2 NEW_LINE N = len ( arr ) NEW_LINE print ( minimumCost ( arr , N , X , Y ) ) NEW_LINE DEDENT
def findMinMax ( a ) : NEW_LINE
min_val = 1000000000 NEW_LINE
for i in range ( 1 , len ( a ) ) : NEW_LINE
min_val = min ( min_val , a [ i ] * a [ i - 1 ] ) NEW_LINE
return min_val NEW_LINE
if __name__ == ( " _ _ main _ _ " ) : NEW_LINE INDENT arr = [ 6 , 4 , 5 , 6 , 2 , 4 , 1 ] NEW_LINE print ( findMinMax ( arr ) ) NEW_LINE DEDENT
sum = 0 NEW_LINE class Node : NEW_LINE
def __init__ ( self , data ) : NEW_LINE INDENT self . data = data NEW_LINE self . left = None NEW_LINE self . right = None NEW_LINE DEDENT
def kDistanceDownSum ( root , k ) : NEW_LINE INDENT global sum NEW_LINE DEDENT
if ( root == None or k < 0 ) : NEW_LINE INDENT return NEW_LINE DEDENT
if ( k == 0 ) : NEW_LINE INDENT sum += root . data NEW_LINE return NEW_LINE DEDENT
kDistanceDownSum ( root . left , k - 1 ) NEW_LINE kDistanceDownSum ( root . right , k - 1 ) NEW_LINE
def kDistanceSum ( root , target , k ) : NEW_LINE INDENT global sum NEW_LINE DEDENT
if ( root == None ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
if ( root . data == target ) : NEW_LINE INDENT kDistanceDownSum ( root . left , k - 1 ) NEW_LINE return 0 NEW_LINE DEDENT
dl = - 1 NEW_LINE
if ( target < root . data ) : NEW_LINE INDENT dl = kDistanceSum ( root . left , target , k ) NEW_LINE DEDENT
if ( dl != - 1 ) : NEW_LINE
if ( dl + 1 == k ) : NEW_LINE INDENT sum += root . data NEW_LINE DEDENT
return - 1 NEW_LINE
dr = - 1 NEW_LINE if ( target > root . data ) : NEW_LINE INDENT dr = kDistanceSum ( root . right , target , k ) NEW_LINE DEDENT if ( dr != - 1 ) : NEW_LINE
if ( dr + 1 == k ) : NEW_LINE INDENT sum += root . data NEW_LINE DEDENT
else : NEW_LINE INDENT kDistanceDownSum ( root . left , k - dr - 2 ) NEW_LINE DEDENT return 1 + dr NEW_LINE
return - 1 NEW_LINE
def insertNode ( data , root ) : NEW_LINE
if ( root == None ) : NEW_LINE INDENT node = Node ( data ) NEW_LINE return node NEW_LINE DEDENT
elif ( data > root . data ) : NEW_LINE INDENT root . right = insertNode ( data , root . right ) NEW_LINE DEDENT
elif ( data <= root . data ) : NEW_LINE INDENT root . left = insertNode ( data , root . left ) NEW_LINE DEDENT
return root NEW_LINE
def findSum ( root , target , K ) : NEW_LINE
kDistanceSum ( root , target , K ) NEW_LINE
print ( sum ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT root = None NEW_LINE N = 11 NEW_LINE tree = [ 3 , 1 , 7 , 0 , 2 , 5 , 10 , 4 , 6 , 9 , 8 ] NEW_LINE DEDENT
for i in range ( N ) : NEW_LINE INDENT root = insertNode ( tree [ i ] , root ) NEW_LINE DEDENT target = 7 NEW_LINE K = 2 NEW_LINE findSum ( root , target , K ) NEW_LINE
def itemType ( n ) : NEW_LINE
count = 0 NEW_LINE
day = 1 NEW_LINE while ( True ) : NEW_LINE
for type in range ( day , 0 , - 1 ) : NEW_LINE INDENT count += type NEW_LINE DEDENT
if ( count >= n ) : NEW_LINE INDENT return type NEW_LINE DEDENT
N = 10 NEW_LINE print ( itemType ( N ) ) NEW_LINE
from math import log2 NEW_LINE
def FindSum ( arr , N ) : NEW_LINE
res = 0 NEW_LINE
for i in range ( N ) : NEW_LINE
power = int ( log2 ( arr [ i ] ) ) NEW_LINE
LesserValue = pow ( 2 , power ) NEW_LINE
LargerValue = pow ( 2 , power + 1 ) NEW_LINE
if ( ( arr [ i ] - LesserValue ) == ( LargerValue - arr [ i ] ) ) : NEW_LINE
res += arr [ i ] NEW_LINE
return res NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 10 , 24 , 17 , 3 , 8 ] NEW_LINE N = len ( arr ) NEW_LINE print ( FindSum ( arr , N ) ) NEW_LINE DEDENT
def findLast ( mat ) : NEW_LINE INDENT m = len ( mat ) NEW_LINE n = len ( mat [ 0 ] ) NEW_LINE DEDENT
rows = set ( ) NEW_LINE cols = set ( ) NEW_LINE for i in range ( m ) : NEW_LINE INDENT for j in range ( n ) : NEW_LINE INDENT if mat [ i ] [ j ] : NEW_LINE INDENT rows . add ( i ) NEW_LINE cols . add ( j ) NEW_LINE DEDENT DEDENT DEDENT
avRows = m - len ( list ( rows ) ) NEW_LINE avCols = n - len ( list ( cols ) ) NEW_LINE
choices = min ( avRows , avCols ) NEW_LINE
if choices & 1 : NEW_LINE
print ( ' P1' ) NEW_LINE
else : NEW_LINE INDENT print ( ' P2' ) NEW_LINE DEDENT
mat = [ [ 1 , 0 , 0 ] , [ 0 , 0 , 0 ] , [ 0 , 0 , 1 ] ] NEW_LINE findLast ( mat ) NEW_LINE
from math import log2 , pow NEW_LINE MOD = 1000000007 NEW_LINE
def sumOfBinaryNumbers ( n ) : NEW_LINE
ans = 0 NEW_LINE one = 1 NEW_LINE
while ( 1 ) : NEW_LINE
if ( n <= 1 ) : NEW_LINE INDENT ans = ( ans + n ) % MOD NEW_LINE break NEW_LINE DEDENT
x = int ( log2 ( n ) ) NEW_LINE cur = 0 NEW_LINE add = ( one << ( x - 1 ) ) NEW_LINE
for i in range ( 1 , x + 1 , 1 ) : NEW_LINE
cur = ( cur + add ) % MOD NEW_LINE add = ( add * 10 % MOD ) NEW_LINE
ans = ( ans + cur ) % MOD NEW_LINE
rem = n - ( one << x ) + 1 NEW_LINE
p = pow ( 10 , x ) NEW_LINE p = ( p * ( rem % MOD ) ) % MOD NEW_LINE ans = ( ans + p ) % MOD NEW_LINE
n = rem - 1 NEW_LINE
print ( int ( ans ) ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 3 NEW_LINE sumOfBinaryNumbers ( N ) NEW_LINE DEDENT
def nearestFibonacci ( num ) : NEW_LINE
if ( num == 0 ) : NEW_LINE INDENT print ( 0 ) NEW_LINE return NEW_LINE DEDENT
first = 0 NEW_LINE second = 1 NEW_LINE
third = first + second NEW_LINE
while ( third <= num ) : NEW_LINE
first = second NEW_LINE
second = third NEW_LINE
third = first + second NEW_LINE
if ( abs ( third - num ) >= abs ( second - num ) ) : NEW_LINE INDENT ans = second NEW_LINE DEDENT else : NEW_LINE INDENT ans = third NEW_LINE DEDENT
print ( ans ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 17 NEW_LINE nearestFibonacci ( N ) NEW_LINE DEDENT
import sys NEW_LINE
def checkPermutation ( ans , a , n ) : NEW_LINE
Max = - sys . maxsize - 1 NEW_LINE
for i in range ( n ) : NEW_LINE
Max = max ( Max , ans [ i ] ) NEW_LINE
if ( Max != a [ i ] ) : NEW_LINE INDENT return False NEW_LINE DEDENT
return True NEW_LINE
def findPermutation ( a , n ) : NEW_LINE
ans = [ 0 ] * n NEW_LINE
um = { } NEW_LINE
for i in range ( n ) : NEW_LINE
if ( a [ i ] not in um ) : NEW_LINE
ans [ i ] = a [ i ] NEW_LINE um [ a [ i ] ] = i NEW_LINE
v = [ ] NEW_LINE j = 0 NEW_LINE
for i in range ( 1 , n + 1 ) : NEW_LINE
if ( i not in um ) : NEW_LINE INDENT v . append ( i ) NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE
if ( ans [ i ] == 0 ) : NEW_LINE INDENT ans [ i ] = v [ j ] NEW_LINE j += 1 NEW_LINE DEDENT
if ( checkPermutation ( ans , a , n ) ) : NEW_LINE
for i in range ( n ) : NEW_LINE INDENT print ( ans [ i ] , end = " ▁ " ) NEW_LINE DEDENT
else : NEW_LINE INDENT print ( " - 1" ) NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 1 , 3 , 4 , 5 , 5 ] NEW_LINE N = len ( arr ) NEW_LINE DEDENT
findPermutation ( arr , N ) NEW_LINE
def countEqualElementPairs ( arr , N ) : NEW_LINE
mp = { } NEW_LINE
for i in range ( N ) : NEW_LINE INDENT if arr [ i ] in mp : NEW_LINE INDENT mp [ arr [ i ] ] += 1 NEW_LINE DEDENT else : NEW_LINE INDENT mp [ arr [ i ] ] = 1 NEW_LINE DEDENT DEDENT
total = 0 NEW_LINE
for key , value in mp . items ( ) : NEW_LINE
total += ( value * ( value - 1 ) ) / 2 NEW_LINE
for i in range ( N ) : NEW_LINE
print ( int ( total - ( mp [ arr [ i ] ] - 1 ) ) , end = " ▁ " ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
arr = [ 1 , 1 , 2 , 1 , 2 ] NEW_LINE
N = len ( arr ) NEW_LINE countEqualElementPairs ( arr , N ) NEW_LINE
def count ( N ) : NEW_LINE INDENT sum = 0 ; NEW_LINE DEDENT
for i in range ( N + 1 ) : NEW_LINE INDENT sum += 7 * ( 8 ** ( i - 1 ) ) ; NEW_LINE DEDENT return int ( sum ) ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT N = 4 ; NEW_LINE print ( count ( N ) ) ; NEW_LINE DEDENT
from math import sqrt ; NEW_LINE
def isPalindrome ( n ) : NEW_LINE
string = str ( n ) ; NEW_LINE
s = 0 ; e = len ( string ) - 1 ; NEW_LINE while ( s < e ) : NEW_LINE
if ( string [ s ] != string [ e ] ) : NEW_LINE INDENT return False ; NEW_LINE DEDENT s += 1 ; NEW_LINE e -= 1 ; NEW_LINE return True ; NEW_LINE
def palindromicDivisors ( n ) : NEW_LINE
PalindromDivisors = [ ] ; NEW_LINE for i in range ( 1 , int ( sqrt ( n ) ) ) : NEW_LINE
if ( n % i == 0 ) : NEW_LINE
if ( n // i == i ) : NEW_LINE
if ( isPalindrome ( i ) ) : NEW_LINE INDENT PalindromDivisors . append ( i ) ; NEW_LINE DEDENT else : NEW_LINE
if ( isPalindrome ( i ) ) : NEW_LINE INDENT PalindromDivisors . append ( i ) ; NEW_LINE DEDENT
if ( isPalindrome ( n // i ) ) : NEW_LINE INDENT PalindromDivisors . append ( n // i ) ; NEW_LINE DEDENT
PalindromDivisors . sort ( ) ; NEW_LINE for i in range ( len ( PalindromDivisors ) ) : NEW_LINE INDENT print ( PalindromDivisors [ i ] , end = " ▁ " ) ; NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 66 ; NEW_LINE DEDENT
palindromicDivisors ( n ) ; NEW_LINE
import sys NEW_LINE
def findMinDel ( arr , n ) : NEW_LINE
min_num = sys . maxsize ; NEW_LINE
for i in range ( n ) : NEW_LINE INDENT min_num = min ( arr [ i ] , min_num ) ; NEW_LINE DEDENT
cnt = 0 ; NEW_LINE
for i in range ( n ) : NEW_LINE INDENT if ( arr [ i ] == min_num ) : NEW_LINE INDENT cnt += 1 ; NEW_LINE DEDENT DEDENT
return n - cnt ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 3 , 3 , 2 ] ; NEW_LINE n = len ( arr ) ; NEW_LINE print ( findMinDel ( arr , n ) ) ; NEW_LINE DEDENT
from math import gcd NEW_LINE
def cntSubArr ( arr , n ) : NEW_LINE
ans = 0 ; NEW_LINE for i in range ( n ) : NEW_LINE
' NEW_LINE INDENT curr_gcd = 0 ; NEW_LINE DEDENT
for j in range ( i , n ) : NEW_LINE INDENT curr_gcd = gcd ( curr_gcd , arr [ j ] ) ; NEW_LINE DEDENT
ans += ( curr_gcd == 1 ) ; NEW_LINE
return ans ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 1 , 1 , 1 ] ; NEW_LINE n = len ( arr ) ; NEW_LINE print ( cntSubArr ( arr , n ) ) ; NEW_LINE DEDENT
def print_primes_till_N ( N ) : NEW_LINE
i , j , flag = 0 , 0 , 0 ; NEW_LINE
print ( " Prime ▁ numbers ▁ between ▁ 1 ▁ and ▁ " , N , " ▁ are : " ) ; NEW_LINE
for i in range ( 1 , N + 1 , 1 ) : NEW_LINE
if ( i == 1 or i == 0 ) : NEW_LINE INDENT continue ; NEW_LINE DEDENT
flag = 1 ; NEW_LINE for j in range ( 2 , ( ( i // 2 ) + 1 ) , 1 ) : NEW_LINE INDENT if ( i % j == 0 ) : NEW_LINE INDENT flag = 0 ; NEW_LINE break ; NEW_LINE DEDENT DEDENT
if ( flag == 1 ) : NEW_LINE INDENT print ( i , end = " ▁ " ) ; NEW_LINE DEDENT
N = 100 ; NEW_LINE print_primes_till_N ( N ) ; NEW_LINE
MAX = 32 NEW_LINE
def findX ( A , B ) : NEW_LINE INDENT X = 0 ; NEW_LINE DEDENT
for bit in range ( MAX ) : NEW_LINE
tempBit = 1 << bit ; NEW_LINE
bitOfX = A & B & tempBit ; NEW_LINE
X += bitOfX ; NEW_LINE return X ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT A = 11 ; B = 13 ; NEW_LINE print ( findX ( A , B ) ) ; NEW_LINE DEDENT
def cntSubSets ( arr , n ) : NEW_LINE
maxVal = max ( arr ) ; NEW_LINE
cnt = 0 ; NEW_LINE for i in range ( n ) : NEW_LINE INDENT if ( arr [ i ] == maxVal ) : NEW_LINE INDENT cnt += 1 ; NEW_LINE DEDENT DEDENT
return ( ( 2 ** cnt ) - 1 ) ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 1 , 2 , 1 , 2 ] ; NEW_LINE n = len ( arr ) ; NEW_LINE print ( cntSubSets ( arr , n ) ) ; NEW_LINE DEDENT
import sys NEW_LINE
def findProb ( arr , n ) : NEW_LINE
maxSum = - ( sys . maxsize - 1 ) ; NEW_LINE maxCount = 0 ; NEW_LINE totalPairs = 0 ; NEW_LINE
for i in range ( n - 1 ) : NEW_LINE INDENT for j in range ( i + 1 , n ) : NEW_LINE DEDENT
sum = arr [ i ] + arr [ j ] ; NEW_LINE
if ( sum == maxSum ) : NEW_LINE
maxCount += 1 ; NEW_LINE
elif ( sum > maxSum ) : NEW_LINE
maxSum = sum ; NEW_LINE maxCount = 1 ; NEW_LINE totalPairs += 1 ; NEW_LINE
prob = maxCount / totalPairs ; NEW_LINE return prob ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 1 , 1 , 1 , 2 , 2 , 2 ] ; NEW_LINE n = len ( arr ) ; NEW_LINE print ( findProb ( arr , n ) ) ; NEW_LINE DEDENT
import math NEW_LINE
def maxCommonFactors ( a , b ) : NEW_LINE
gcd = math . gcd ( a , b ) NEW_LINE
ans = 1 ; NEW_LINE
i = 2 NEW_LINE while ( i * i <= gcd ) : NEW_LINE INDENT if ( gcd % i == 0 ) : NEW_LINE INDENT ans += 1 NEW_LINE while ( gcd % i == 0 ) : NEW_LINE INDENT gcd = gcd // i NEW_LINE DEDENT DEDENT i += 1 NEW_LINE DEDENT
if ( gcd != 1 ) : NEW_LINE INDENT ans += 1 NEW_LINE DEDENT
return ans NEW_LINE
a = 12 NEW_LINE b = 18 NEW_LINE print ( maxCommonFactors ( a , b ) ) NEW_LINE
days = [ 31 , 28 , 31 , 30 , 31 , 30 , 31 , 31 , 30 , 31 , 30 , 31 ] ; NEW_LINE
def dayOfYear ( date ) : NEW_LINE
year = ( int ) ( date [ 0 : 4 ] ) ; NEW_LINE month = ( int ) ( date [ 5 : 7 ] ) ; NEW_LINE day = ( int ) ( date [ 8 : ] ) ; NEW_LINE
if ( month > 2 and year % 4 == 0 and ( year % 100 != 0 or year % 400 == 0 ) ) : NEW_LINE INDENT day += 1 ; NEW_LINE DEDENT
month -= 1 ; NEW_LINE while ( month > 0 ) : NEW_LINE INDENT day = day + days [ month - 1 ] ; NEW_LINE month -= 1 ; NEW_LINE DEDENT return day ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT date = "2019-01-09" ; NEW_LINE print ( dayOfYear ( date ) ) ; NEW_LINE DEDENT
def Cells ( n , x ) : NEW_LINE INDENT ans = 0 ; NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE INDENT if ( x % i == 0 and x / i <= n ) : NEW_LINE INDENT ans += 1 ; NEW_LINE DEDENT DEDENT return ans ; NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 6 ; x = 12 ; NEW_LINE DEDENT
print ( Cells ( n , x ) ) ; NEW_LINE
import math NEW_LINE
def nextPowerOfFour ( n ) : NEW_LINE INDENT x = math . floor ( ( n ** ( 1 / 2 ) ) ** ( 1 / 2 ) ) ; NEW_LINE DEDENT
if ( ( x ** 4 ) == n ) : NEW_LINE INDENT return n ; NEW_LINE DEDENT else : NEW_LINE INDENT x = x + 1 ; NEW_LINE return ( x ** 4 ) ; NEW_LINE DEDENT
n = 122 ; NEW_LINE print ( nextPowerOfFour ( n ) ) ; NEW_LINE
def minOperations ( x , y , p , q ) : NEW_LINE
if ( y % x != 0 ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT d = y // x NEW_LINE
a = 0 NEW_LINE
while ( d % p == 0 ) : NEW_LINE INDENT d //= p NEW_LINE a += 1 NEW_LINE DEDENT
b = 0 NEW_LINE
while ( d % q == 0 ) : NEW_LINE INDENT d //= q NEW_LINE b += 1 NEW_LINE DEDENT
if ( d != 1 ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
return ( a + b ) NEW_LINE
x = 12 NEW_LINE y = 2592 NEW_LINE p = 2 NEW_LINE q = 3 NEW_LINE print ( minOperations ( x , y , p , q ) ) NEW_LINE
from math import sqrt NEW_LINE
def nCr ( n ) : NEW_LINE
if ( n < 4 ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT answer = n * ( n - 1 ) * ( n - 2 ) * ( n - 3 ) ; NEW_LINE answer //= 24 ; NEW_LINE return answer ; NEW_LINE
def countQuadruples ( N , K ) : NEW_LINE
M = N // K ; NEW_LINE answer = nCr ( M ) ; NEW_LINE
for i in range ( 2 , M ) : NEW_LINE INDENT j = i ; NEW_LINE DEDENT
temp2 = M // i ; NEW_LINE
count = 0 ; NEW_LINE
check = 0 ; NEW_LINE temp = j ; NEW_LINE while ( j % 2 == 0 ) : NEW_LINE INDENT count += 1 ; NEW_LINE j //= 2 ; NEW_LINE if ( count >= 2 ) : NEW_LINE INDENT break ; NEW_LINE DEDENT DEDENT if ( count >= 2 ) : NEW_LINE INDENT check = 1 ; NEW_LINE DEDENT for k in range ( 3 , int ( sqrt ( temp ) ) , 2 ) : NEW_LINE INDENT cnt = 0 ; NEW_LINE while ( j % k == 0 ) : NEW_LINE INDENT cnt += 1 ; NEW_LINE j //= k ; NEW_LINE if ( cnt >= 2 ) : NEW_LINE INDENT break ; NEW_LINE DEDENT DEDENT if ( cnt >= 2 ) : NEW_LINE INDENT check = 1 ; NEW_LINE break ; NEW_LINE DEDENT elif ( cnt == 1 ) : NEW_LINE INDENT count += 1 ; NEW_LINE DEDENT DEDENT if ( j > 2 ) : NEW_LINE INDENT count += 1 ; NEW_LINE DEDENT
if ( check ) : NEW_LINE INDENT continue ; NEW_LINE DEDENT else : NEW_LINE
if ( count % 2 == 1 ) : NEW_LINE INDENT answer -= nCr ( temp2 ) ; NEW_LINE DEDENT else : NEW_LINE INDENT answer += nCr ( temp2 ) ; NEW_LINE DEDENT return answer ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT N = 10 ; K = 2 ; NEW_LINE print ( countQuadruples ( N , K ) ) ; NEW_LINE DEDENT
def getX ( a , b , c , d ) : NEW_LINE INDENT X = ( b * c - a * d ) // ( d - c ) NEW_LINE return X NEW_LINE DEDENT
a = 2 NEW_LINE b = 3 NEW_LINE c = 4 NEW_LINE d = 5 NEW_LINE print ( getX ( a , b , c , d ) ) NEW_LINE
def isVowel ( ch ) : NEW_LINE INDENT if ( ch == ' a ' or ch == ' e ' or ch == ' i ' or ch == ' o ' or ch == ' u ' ) : NEW_LINE INDENT return True NEW_LINE DEDENT else : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT
def fact ( n ) : NEW_LINE INDENT if ( n < 2 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT return n * fact ( n - 1 ) NEW_LINE DEDENT
def only_vowels ( freq ) : NEW_LINE INDENT denom = 1 NEW_LINE cnt_vwl = 0 NEW_LINE DEDENT
for itr in freq : NEW_LINE INDENT if ( isVowel ( itr ) ) : NEW_LINE INDENT denom *= fact ( freq [ itr ] ) NEW_LINE cnt_vwl += freq [ itr ] NEW_LINE DEDENT DEDENT return fact ( cnt_vwl ) // denom NEW_LINE
def all_vowels_together ( freq ) : NEW_LINE
vow = only_vowels ( freq ) NEW_LINE
denom = 1 NEW_LINE
cnt_cnst = 0 NEW_LINE for itr in freq : NEW_LINE INDENT if ( isVowel ( itr ) == False ) : NEW_LINE INDENT denom *= fact ( freq [ itr ] ) NEW_LINE cnt_cnst += freq [ itr ] NEW_LINE DEDENT DEDENT
ans = fact ( cnt_cnst + 1 ) // denom NEW_LINE return ( ans * vow ) NEW_LINE
def total_permutations ( freq ) : NEW_LINE
cnt = 0 NEW_LINE
denom = 1 NEW_LINE for itr in freq : NEW_LINE INDENT denom *= fact ( freq [ itr ] ) NEW_LINE cnt += freq [ itr ] NEW_LINE DEDENT
return fact ( cnt ) // denom NEW_LINE
def no_vowels_together ( word ) : NEW_LINE
freq = dict ( ) NEW_LINE
for i in word : NEW_LINE INDENT ch = i . lower ( ) NEW_LINE freq [ ch ] = freq . get ( ch , 0 ) + 1 NEW_LINE DEDENT
total = total_permutations ( freq ) NEW_LINE
vwl_tgthr = all_vowels_together ( freq ) NEW_LINE
res = total - vwl_tgthr NEW_LINE
return res NEW_LINE
word = " allahabad " NEW_LINE ans = no_vowels_together ( word ) NEW_LINE print ( ans ) NEW_LINE word = " geeksforgeeks " NEW_LINE ans = no_vowels_together ( word ) NEW_LINE print ( ans ) NEW_LINE word = " abcd " NEW_LINE ans = no_vowels_together ( word ) NEW_LINE print ( ans ) NEW_LINE
def numberOfMen ( D , m , d ) : NEW_LINE INDENT Men = ( m * ( D - d ) ) / d ; NEW_LINE return int ( Men ) ; NEW_LINE DEDENT
D = 5 ; m = 4 ; d = 4 ; NEW_LINE print ( numberOfMen ( D , m , d ) ) ; NEW_LINE
def area ( a , b , c ) : NEW_LINE INDENT d = abs ( ( c * c ) / ( 2 * a * b ) ) NEW_LINE return d NEW_LINE DEDENT
a = - 2 NEW_LINE b = 4 NEW_LINE c = 3 NEW_LINE print ( area ( a , b , c ) ) NEW_LINE
def addToArrayForm ( A , K ) : NEW_LINE
v , ans = [ ] , [ ] NEW_LINE
rem , i = 0 , 0 NEW_LINE
for i in range ( len ( A ) - 1 , - 1 , - 1 ) : NEW_LINE
my = A [ i ] + ( K % 10 ) + rem NEW_LINE if my > 9 : NEW_LINE
rem = 1 NEW_LINE
v . append ( my % 10 ) NEW_LINE else : NEW_LINE v . append ( my ) NEW_LINE rem = 0 NEW_LINE K = K // 10 NEW_LINE
while K > 0 : NEW_LINE
my = ( K % 10 ) + rem NEW_LINE v . append ( my % 10 ) NEW_LINE
if my // 10 > 0 : NEW_LINE INDENT rem = 1 NEW_LINE DEDENT else : NEW_LINE INDENT rem = 0 NEW_LINE DEDENT K = K // 10 NEW_LINE if rem > 0 : NEW_LINE v . append ( rem ) NEW_LINE
for i in range ( len ( v ) - 1 , - 1 , - 1 ) : NEW_LINE INDENT ans . append ( v [ i ] ) NEW_LINE DEDENT return ans NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT A = [ 2 , 7 , 4 ] NEW_LINE K = 181 NEW_LINE ans = addToArrayForm ( A , K ) NEW_LINE DEDENT
for i in range ( 0 , len ( ans ) ) : NEW_LINE INDENT print ( ans [ i ] , end = " " ) NEW_LINE DEDENT
MAX = 100005 ; NEW_LINE
def kadaneAlgorithm ( ar , n ) : NEW_LINE INDENT sum = 0 ; maxSum = 0 ; NEW_LINE for i in range ( n ) : NEW_LINE INDENT sum += ar [ i ] ; NEW_LINE if ( sum < 0 ) : NEW_LINE INDENT sum = 0 ; NEW_LINE DEDENT maxSum = max ( maxSum , sum ) ; NEW_LINE DEDENT return maxSum ; NEW_LINE DEDENT
def maxFunction ( arr , n ) : NEW_LINE INDENT b = [ 0 ] * MAX ; NEW_LINE c = [ 0 ] * MAX ; NEW_LINE DEDENT
for i in range ( n - 1 ) : NEW_LINE INDENT if ( i & 1 ) : NEW_LINE INDENT b [ i ] = abs ( arr [ i + 1 ] - arr [ i ] ) ; NEW_LINE c [ i ] = - b [ i ] ; NEW_LINE DEDENT else : NEW_LINE INDENT c [ i ] = abs ( arr [ i + 1 ] - arr [ i ] ) ; NEW_LINE b [ i ] = - c [ i ] ; NEW_LINE DEDENT DEDENT
ans = kadaneAlgorithm ( b , n - 1 ) ; NEW_LINE ans = max ( ans , kadaneAlgorithm ( c , n - 1 ) ) ; NEW_LINE return ans ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 1 , 5 , 4 , 7 ] ; NEW_LINE n = len ( arr ) NEW_LINE print ( maxFunction ( arr , n ) ) ; NEW_LINE DEDENT
def findThirdDigit ( n ) : NEW_LINE
if n < 3 : NEW_LINE INDENT return 0 NEW_LINE DEDENT
return 1 if n and 1 else 6 NEW_LINE
n = 7 NEW_LINE print ( findThirdDigit ( n ) ) NEW_LINE
def getProbability ( a , b , c , d ) : NEW_LINE
p = a / b ; NEW_LINE q = c / d ; NEW_LINE
ans = p * ( 1 / ( 1 - ( 1 - q ) * ( 1 - p ) ) ) ; NEW_LINE return round ( ans , 5 ) ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = 1 ; b = 2 ; c = 10 ; d = 11 ; NEW_LINE print ( getProbability ( a , b , c , d ) ) ; NEW_LINE DEDENT
def isPalindrome ( n ) : NEW_LINE
divisor = 1 NEW_LINE while ( int ( n / divisor ) >= 10 ) : NEW_LINE INDENT divisor *= 10 NEW_LINE DEDENT while ( n != 0 ) : NEW_LINE INDENT leading = int ( n / divisor ) NEW_LINE trailing = n % 10 NEW_LINE DEDENT
if ( leading != trailing ) : NEW_LINE INDENT return False NEW_LINE DEDENT
n = int ( ( n % divisor ) / 10 ) NEW_LINE
divisor = int ( divisor / 100 ) NEW_LINE return True NEW_LINE
def largestPalindrome ( A , n ) : NEW_LINE INDENT currentMax = - 1 NEW_LINE for i in range ( 0 , n , 1 ) : NEW_LINE DEDENT
if ( A [ i ] > currentMax and isPalindrome ( A [ i ] ) ) : NEW_LINE INDENT currentMax = A [ i ] NEW_LINE DEDENT
return currentMax NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT A = [ 1 , 232 , 54545 , 999991 ] NEW_LINE n = len ( A ) NEW_LINE DEDENT
print ( largestPalindrome ( A , n ) ) NEW_LINE
def getFinalElement ( n ) : NEW_LINE INDENT finalNum = 2 NEW_LINE while finalNum * 2 <= n : NEW_LINE INDENT finalNum *= 2 NEW_LINE DEDENT return finalNum NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT N = 12 NEW_LINE print ( getFinalElement ( N ) ) NEW_LINE DEDENT
import math as mt NEW_LINE
def SieveOfEratosthenes ( prime , p_size ) : NEW_LINE
prime [ 0 ] = False NEW_LINE prime [ 1 ] = False NEW_LINE for p in range ( 2 , mt . ceil ( mt . sqrt ( p_size + 1 ) ) ) : NEW_LINE
if ( prime [ p ] ) : NEW_LINE
for i in range ( p * 2 , p_size + 1 , p ) : NEW_LINE INDENT prime [ i ] = False NEW_LINE DEDENT
def SumOfElements ( arr , n ) : NEW_LINE INDENT prime = [ True for i in range ( n + 1 ) ] NEW_LINE SieveOfEratosthenes ( prime , n + 1 ) NEW_LINE i , j = 0 , 0 NEW_LINE DEDENT
m = dict ( ) NEW_LINE for i in range ( n ) : NEW_LINE INDENT if arr [ i ] in m . keys ( ) : NEW_LINE INDENT m [ arr [ i ] ] += 1 NEW_LINE DEDENT else : NEW_LINE INDENT m [ arr [ i ] ] = 1 NEW_LINE DEDENT DEDENT Sum = 0 NEW_LINE
for i in m : NEW_LINE
if ( prime [ m [ i ] ] ) : NEW_LINE INDENT Sum += ( i ) NEW_LINE DEDENT return Sum NEW_LINE
arr = [ 5 , 4 , 6 , 5 , 4 , 6 ] NEW_LINE n = len ( arr ) NEW_LINE print ( SumOfElements ( arr , n ) ) NEW_LINE
def isPalindrome ( num ) : NEW_LINE INDENT reverse_num = 0 NEW_LINE DEDENT
temp = num NEW_LINE while ( temp != 0 ) : NEW_LINE INDENT remainder = temp % 10 NEW_LINE reverse_num = reverse_num * 10 + remainder NEW_LINE temp = int ( temp / 10 ) NEW_LINE DEDENT
if ( reverse_num == num ) : NEW_LINE INDENT return True NEW_LINE DEDENT return False NEW_LINE
def isOddLength ( num ) : NEW_LINE INDENT count = 0 NEW_LINE while ( num > 0 ) : NEW_LINE INDENT num = int ( num / 10 ) NEW_LINE count += 1 NEW_LINE DEDENT if ( count % 2 != 0 ) : NEW_LINE INDENT return True NEW_LINE DEDENT return False NEW_LINE DEDENT
def sumOfAllPalindrome ( L , R ) : NEW_LINE INDENT sum = 0 NEW_LINE if ( L <= R ) : NEW_LINE INDENT for i in range ( L , R + 1 , 1 ) : NEW_LINE DEDENT DEDENT
if ( isPalindrome ( i ) and isOddLength ( i ) ) : NEW_LINE INDENT sum += i NEW_LINE DEDENT return sum NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT L = 110 NEW_LINE R = 1130 NEW_LINE print ( sumOfAllPalindrome ( L , R ) ) NEW_LINE DEDENT
def fact ( n ) : NEW_LINE INDENT f = 1 NEW_LINE for i in range ( 2 , n + 1 ) : NEW_LINE INDENT f = f * i NEW_LINE DEDENT return f NEW_LINE DEDENT
def waysOfConsonants ( size1 , freq ) : NEW_LINE INDENT ans = fact ( size1 ) NEW_LINE for i in range ( 26 ) : NEW_LINE DEDENT
if ( i == 0 or i == 4 or i == 8 or i == 14 or i == 20 ) : NEW_LINE INDENT continue NEW_LINE DEDENT else : NEW_LINE INDENT ans = ans // fact ( freq [ i ] ) NEW_LINE DEDENT return ans NEW_LINE
def waysOfVowels ( size2 , freq ) : NEW_LINE INDENT return ( fact ( size2 ) // ( fact ( freq [ 0 ] ) * fact ( freq [ 4 ] ) * fact ( freq [ 8 ] ) * fact ( freq [ 14 ] ) * fact ( freq [ 20 ] ) ) ) NEW_LINE DEDENT
def countWays ( str1 ) : NEW_LINE INDENT freq = [ 0 ] * 26 NEW_LINE for i in range ( len ( str1 ) ) : NEW_LINE INDENT freq [ ord ( str1 [ i ] ) - ord ( ' a ' ) ] += 1 NEW_LINE DEDENT DEDENT
vowel = 0 NEW_LINE consonant = 0 NEW_LINE for i in range ( len ( str1 ) ) : NEW_LINE INDENT if ( str1 [ i ] != ' a ' and str1 [ i ] != ' e ' and str1 [ i ] != ' i ' and str1 [ i ] != ' o ' and str1 [ i ] != ' u ' ) : NEW_LINE INDENT consonant += 1 NEW_LINE DEDENT else : NEW_LINE INDENT vowel += 1 NEW_LINE DEDENT DEDENT
return ( waysOfConsonants ( consonant + 1 , freq ) * waysOfVowels ( vowel , freq ) ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT str1 = " geeksforgeeks " NEW_LINE print ( countWays ( str1 ) ) NEW_LINE DEDENT
def calculateAlternateSum ( n ) : NEW_LINE INDENT if ( n <= 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT fibo = [ 0 ] * ( n + 1 ) NEW_LINE fibo [ 0 ] = 0 NEW_LINE fibo [ 1 ] = 1 NEW_LINE DEDENT
sum = pow ( fibo [ 0 ] , 2 ) + pow ( fibo [ 1 ] , 2 ) NEW_LINE
for i in range ( 2 , n + 1 ) : NEW_LINE INDENT fibo [ i ] = fibo [ i - 1 ] + fibo [ i - 2 ] NEW_LINE DEDENT
if ( i % 2 == 0 ) : NEW_LINE INDENT sum -= fibo [ i ] NEW_LINE DEDENT
else : NEW_LINE INDENT sum += fibo [ i ] NEW_LINE DEDENT
return sum NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
n = 8 NEW_LINE
print ( " Alternating ▁ Fibonacci ▁ Sum ▁ upto ▁ " , n , " ▁ terms : ▁ " , calculateAlternateSum ( n ) ) NEW_LINE
def getValue ( n ) : NEW_LINE INDENT i = 0 ; NEW_LINE k = 1 ; NEW_LINE while ( i < n ) : NEW_LINE INDENT i = i + k ; NEW_LINE k = k * 2 ; NEW_LINE DEDENT return int ( k / 2 ) ; NEW_LINE DEDENT
n = 9 ; NEW_LINE
print ( getValue ( n ) ) ; NEW_LINE
n = 1025 ; NEW_LINE
print ( getValue ( n ) ) ; NEW_LINE
import math NEW_LINE
def countDigits ( val , arr ) : NEW_LINE INDENT while ( val > 0 ) : NEW_LINE INDENT digit = val % 10 NEW_LINE arr [ int ( digit ) ] += 1 NEW_LINE val = val // 10 NEW_LINE DEDENT return ; NEW_LINE DEDENT def countFrequency ( x , n ) : NEW_LINE
freq_count = [ 0 ] * 10 NEW_LINE
for i in range ( 1 , n + 1 ) : NEW_LINE
val = math . pow ( x , i ) NEW_LINE
countDigits ( val , freq_count ) NEW_LINE
for i in range ( 10 ) : NEW_LINE INDENT print ( freq_count [ i ] , end = " ▁ " ) ; NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT x = 15 NEW_LINE n = 3 NEW_LINE countFrequency ( x , n ) NEW_LINE DEDENT
def countSolutions ( a ) : NEW_LINE INDENT count = 0 NEW_LINE DEDENT
for i in range ( a + 1 ) : NEW_LINE INDENT if ( a == ( i + ( a ^ i ) ) ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT DEDENT return count NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = 3 NEW_LINE print ( countSolutions ( a ) ) NEW_LINE DEDENT
def countSolutions ( a ) : NEW_LINE INDENT count = bin ( a ) . count ( '1' ) NEW_LINE return 2 ** count NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = 3 NEW_LINE print ( countSolutions ( a ) ) NEW_LINE DEDENT
def calculateAreaSum ( l , b ) : NEW_LINE INDENT size = 1 NEW_LINE DEDENT
maxSize = min ( l , b ) NEW_LINE totalArea = 0 NEW_LINE for i in range ( 1 , maxSize + 1 ) : NEW_LINE
totalSquares = ( ( l - size + 1 ) * ( b - size + 1 ) ) NEW_LINE
area = ( totalSquares * size * size ) NEW_LINE
totalArea += area NEW_LINE
size += 1 NEW_LINE return totalArea NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT l = 4 NEW_LINE b = 3 NEW_LINE print ( calculateAreaSum ( l , b ) ) NEW_LINE DEDENT
def boost_hyperfactorial ( num ) : NEW_LINE
val = 1 ; NEW_LINE for i in range ( 1 , num + 1 ) : NEW_LINE INDENT val = val * pow ( i , i ) ; NEW_LINE DEDENT
return val ; NEW_LINE
num = 5 ; NEW_LINE print ( boost_hyperfactorial ( num ) ) ; NEW_LINE
def boost_hyperfactorial ( num ) : NEW_LINE
val = 1 ; NEW_LINE for i in range ( 1 , num + 1 ) : NEW_LINE INDENT for j in range ( 1 , i + 1 ) : NEW_LINE DEDENT
val *= i ; NEW_LINE
return val ; NEW_LINE
num = 5 ; NEW_LINE print ( boost_hyperfactorial ( num ) ) ; NEW_LINE
def subtractOne ( x ) : NEW_LINE INDENT m = 1 NEW_LINE DEDENT
while ( ( x & m ) == False ) : NEW_LINE INDENT x = x ^ m NEW_LINE m = m << 1 NEW_LINE DEDENT
x = x ^ m NEW_LINE return x NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT print ( subtractOne ( 13 ) ) NEW_LINE DEDENT
rows = 3 ; NEW_LINE cols = 3 ; NEW_LINE
def meanVector ( mat ) : NEW_LINE INDENT print ( " [ ▁ " , end = " " ) ; NEW_LINE DEDENT
for i in range ( rows ) : NEW_LINE
mean = 0.00 ; NEW_LINE
sum = 0 ; NEW_LINE for j in range ( cols ) : NEW_LINE INDENT sum = sum + mat [ j ] [ i ] ; mean = int ( sum / rows ) ; print ( mean , end = " ▁ " ) ; print ( " ] " ) ; NEW_LINE DEDENT
mat = [ [ 1 , 2 , 3 ] , [ 4 , 5 , 6 ] , [ 7 , 8 , 9 ] ] ; NEW_LINE meanVector ( mat ) ; NEW_LINE
import math NEW_LINE
def primeFactors ( n ) : NEW_LINE INDENT res = [ ] NEW_LINE if ( n % 2 == 0 ) : NEW_LINE INDENT while ( n % 2 == 0 ) : NEW_LINE INDENT n = int ( n / 2 ) NEW_LINE DEDENT res . append ( 2 ) NEW_LINE DEDENT DEDENT
for i in range ( 3 , int ( math . sqrt ( n ) ) , 2 ) : NEW_LINE
if ( n % i == 0 ) : NEW_LINE INDENT while ( n % i == 0 ) : NEW_LINE INDENT n = int ( n / i ) NEW_LINE DEDENT res . append ( i ) NEW_LINE DEDENT
if ( n > 2 ) : NEW_LINE INDENT res . append ( n ) NEW_LINE DEDENT return res NEW_LINE
def isHoax ( n ) : NEW_LINE
pf = primeFactors ( n ) NEW_LINE
if ( pf [ 0 ] == n ) : NEW_LINE INDENT return False NEW_LINE DEDENT
all_pf_sum = 0 NEW_LINE for i in range ( 0 , len ( pf ) ) : NEW_LINE
pf_sum = 0 NEW_LINE while ( pf [ i ] > 0 ) : NEW_LINE INDENT pf_sum += pf [ i ] % 10 NEW_LINE pf [ i ] = int ( pf [ i ] / 10 ) NEW_LINE DEDENT all_pf_sum += pf_sum NEW_LINE
sum_n = 0 ; NEW_LINE while ( n > 0 ) : NEW_LINE INDENT sum_n += n % 10 NEW_LINE n = int ( n / 10 ) NEW_LINE DEDENT
return sum_n == all_pf_sum NEW_LINE
n = 84 ; NEW_LINE if ( isHoax ( n ) ) : NEW_LINE INDENT print ( " A Hoax Number " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Not a Hoax Number " ) NEW_LINE DEDENT
def LucasLehmer ( n ) : NEW_LINE
INDENT current_val = 4 ; NEW_LINE DEDENT
INDENT series = [ ] NEW_LINE DEDENT
INDENT series . append ( current_val ) NEW_LINE for i in range ( n ) : NEW_LINE INDENT current_val = current_val * current_val - 2 ; NEW_LINE series . append ( current_val ) ; NEW_LINE DEDENT DEDENT
INDENT for i in range ( n + 1 ) : NEW_LINE INDENT print ( " Term " , i , " : " , series [ i ] ) NEW_LINE DEDENT DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 5 ; NEW_LINE LucasLehmer ( n ) ; NEW_LINE DEDENT
' NEW_LINE def modInverse ( a , prime ) : NEW_LINE INDENT a = a % prime NEW_LINE for x in range ( 1 , prime ) : NEW_LINE INDENT if ( ( a * x ) % prime == 1 ) : NEW_LINE INDENT return x NEW_LINE DEDENT DEDENT return - 1 NEW_LINE DEDENT def printModIverses ( n , prime ) : NEW_LINE INDENT for i in range ( 1 , n + 1 ) : NEW_LINE INDENT print ( modInverse ( i , prime ) , end = " ▁ " ) NEW_LINE DEDENT DEDENT
n = 10 NEW_LINE prime = 17 NEW_LINE printModIverses ( n , prime ) NEW_LINE
def minOp ( num ) : NEW_LINE
count = 0 NEW_LINE
while ( num ) : NEW_LINE INDENT rem = num % 10 NEW_LINE if ( not ( rem == 3 or rem == 8 ) ) : NEW_LINE INDENT count = count + 1 NEW_LINE DEDENT num = num // 10 NEW_LINE DEDENT return count NEW_LINE
num = 234198 NEW_LINE print ( " Minimum ▁ Operations ▁ = " , minOp ( num ) ) NEW_LINE
def sumOfDigits ( a ) : NEW_LINE INDENT sm = 0 NEW_LINE while ( a != 0 ) : NEW_LINE INDENT sm = sm + a % 10 NEW_LINE a = a // 10 NEW_LINE DEDENT return sm NEW_LINE DEDENT
def findMax ( x ) : NEW_LINE
b = 1 NEW_LINE ans = x NEW_LINE
while ( x != 0 ) : NEW_LINE
cur = ( x - 1 ) * b + ( b - 1 ) NEW_LINE
if ( sumOfDigits ( cur ) > sumOfDigits ( ans ) or ( sumOfDigits ( cur ) == sumOfDigits ( ans ) and cur > ans ) ) : NEW_LINE INDENT ans = cur NEW_LINE DEDENT
x = x // 10 NEW_LINE b = b * 10 NEW_LINE return ans NEW_LINE
n = 521 NEW_LINE print ( findMax ( n ) ) NEW_LINE
def median ( a , l , r ) : NEW_LINE INDENT n = r - l + 1 NEW_LINE n = ( n + 1 ) // 2 - 1 NEW_LINE return n + l NEW_LINE DEDENT
def IQR ( a , n ) : NEW_LINE INDENT a . sort ( ) NEW_LINE DEDENT
mid_index = median ( a , 0 , n ) NEW_LINE
Q1 = a [ median ( a , 0 , mid_index ) ] NEW_LINE
Q3 = a [ mid_index + median ( a , mid_index + 1 , n ) ] NEW_LINE
return ( Q3 - Q1 ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT a = [ 1 , 19 , 7 , 6 , 5 , 9 , 12 , 27 , 18 , 2 , 15 ] NEW_LINE n = len ( a ) NEW_LINE print ( IQR ( a , n ) ) NEW_LINE DEDENT
def isPalindrome ( n ) : NEW_LINE
divisor = 1 NEW_LINE while ( n / divisor >= 10 ) : NEW_LINE INDENT divisor *= 10 NEW_LINE DEDENT while ( n != 0 ) : NEW_LINE INDENT leading = n // divisor NEW_LINE trailing = n % 10 NEW_LINE DEDENT
if ( leading != trailing ) : NEW_LINE INDENT return False NEW_LINE DEDENT
n = ( n % divisor ) // 10 NEW_LINE
divisor = divisor // 100 NEW_LINE return True NEW_LINE
def largestPalindrome ( A , n ) : NEW_LINE
A . sort ( ) NEW_LINE for i in range ( n - 1 , - 1 , - 1 ) : NEW_LINE
if ( isPalindrome ( A [ i ] ) ) : NEW_LINE INDENT return A [ i ] NEW_LINE DEDENT
return - 1 NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT A = [ 1 , 232 , 54545 , 999991 ] NEW_LINE n = len ( A ) NEW_LINE DEDENT
print ( largestPalindrome ( A , n ) ) NEW_LINE
def findSum ( n , a , b ) : NEW_LINE INDENT sum = 0 NEW_LINE for i in range ( 0 , n , 1 ) : NEW_LINE DEDENT
if ( i % a == 0 or i % b == 0 ) : NEW_LINE INDENT sum += i NEW_LINE DEDENT return sum NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 10 NEW_LINE a = 3 NEW_LINE b = 5 NEW_LINE print ( findSum ( n , a , b ) ) NEW_LINE DEDENT
def subtractOne ( x ) : NEW_LINE INDENT return ( ( x << 1 ) + ( ~ x ) ) ; NEW_LINE DEDENT print ( subtractOne ( 13 ) ) ; NEW_LINE
def pell ( n ) : NEW_LINE INDENT if ( n <= 2 ) : NEW_LINE INDENT return n NEW_LINE DEDENT return ( 2 * pell ( n - 1 ) + pell ( n - 2 ) ) NEW_LINE DEDENT
n = 4 ; NEW_LINE print ( pell ( n ) ) NEW_LINE
def LCM ( arr , n ) : NEW_LINE
max_num = 0 ; NEW_LINE for i in range ( n ) : NEW_LINE INDENT if ( max_num < arr [ i ] ) : NEW_LINE INDENT max_num = arr [ i ] ; NEW_LINE DEDENT DEDENT
res = 1 ; NEW_LINE
while ( x <= max_num ) : NEW_LINE
indexes = [ ] ; NEW_LINE for j in range ( n ) : NEW_LINE INDENT if ( arr [ j ] % x == 0 ) : NEW_LINE INDENT indexes . append ( j ) ; NEW_LINE DEDENT DEDENT
if ( len ( indexes ) >= 2 ) : NEW_LINE
for j in range ( len ( indexes ) ) : NEW_LINE INDENT arr [ indexes [ j ] ] = int ( arr [ indexes [ j ] ] / x ) ; NEW_LINE DEDENT res = res * x ; NEW_LINE else : NEW_LINE x += 1 ; NEW_LINE
for i in range ( n ) : NEW_LINE INDENT res = res * arr [ i ] ; NEW_LINE DEDENT return res ; NEW_LINE
arr = [ 1 , 2 , 3 , 4 , 5 , 10 , 20 , 35 ] ; NEW_LINE n = len ( arr ) ; NEW_LINE print ( LCM ( arr , n ) ) ; NEW_LINE
import math NEW_LINE
def politness ( n ) : NEW_LINE INDENT count = 0 NEW_LINE DEDENT
for i in range ( 2 , int ( math . sqrt ( 2 * n ) ) + 1 ) : NEW_LINE INDENT if ( ( 2 * n ) % i != 0 ) : NEW_LINE INDENT continue NEW_LINE DEDENT a = 2 * n NEW_LINE a = a / i NEW_LINE a = a - ( i - 1 ) NEW_LINE if ( a % 2 != 0 ) : NEW_LINE INDENT continue NEW_LINE DEDENT a /= 2 NEW_LINE if ( a > 0 ) : NEW_LINE INDENT count = count + 1 NEW_LINE DEDENT DEDENT return count NEW_LINE
n = 90 NEW_LINE print " Politness ▁ of ▁ " , n , " ▁ = ▁ " , politness ( n ) NEW_LINE n = 15 NEW_LINE print " Politness ▁ of ▁ " , n , " ▁ = ▁ " , politness ( n ) NEW_LINE
import math NEW_LINE MAX = 10000 ; NEW_LINE
primes = [ ] ; NEW_LINE
def sieveSundaram ( ) : NEW_LINE
marked = [ False ] * ( int ( MAX / 2 ) + 100 ) ; NEW_LINE
for i in range ( 1 , int ( ( math . sqrt ( MAX ) - 1 ) / 2 ) + 1 ) : NEW_LINE INDENT for j in range ( ( i * ( i + 1 ) ) << 1 , int ( MAX / 2 ) + 1 , 2 * i + 1 ) : NEW_LINE INDENT marked [ j ] = True ; NEW_LINE DEDENT DEDENT
primes . append ( 2 ) ; NEW_LINE
for i in range ( 1 , int ( MAX / 2 ) + 1 ) : NEW_LINE INDENT if ( marked [ i ] == False ) : NEW_LINE INDENT primes . append ( 2 * i + 1 ) ; NEW_LINE DEDENT DEDENT
def findPrimes ( n ) : NEW_LINE
if ( n <= 2 or n % 2 != 0 ) : NEW_LINE INDENT print ( " Invalid ▁ Input " ) ; NEW_LINE return ; NEW_LINE DEDENT
i = 0 ; NEW_LINE while ( primes [ i ] <= n // 2 ) : NEW_LINE
diff = n - primes [ i ] ; NEW_LINE
if diff in primes : NEW_LINE
print ( primes [ i ] , " + " , diff , " = " , n ) ; NEW_LINE return ; NEW_LINE i += 1 ; NEW_LINE
sieveSundaram ( ) ; NEW_LINE
findPrimes ( 4 ) ; NEW_LINE findPrimes ( 38 ) ; NEW_LINE findPrimes ( 100 ) ; NEW_LINE
import math NEW_LINE
def kPrimeFactor ( n , k ) : NEW_LINE
while ( n % 2 == 0 ) : NEW_LINE INDENT k = k - 1 NEW_LINE n = n / 2 NEW_LINE if ( k == 0 ) : NEW_LINE INDENT return 2 NEW_LINE DEDENT DEDENT
i = 3 NEW_LINE while i <= math . sqrt ( n ) : NEW_LINE
while ( n % i == 0 ) : NEW_LINE INDENT if ( k == 1 ) : NEW_LINE INDENT return i NEW_LINE DEDENT k = k - 1 NEW_LINE n = n / i NEW_LINE DEDENT i = i + 2 NEW_LINE
if ( n > 2 and k == 1 ) : NEW_LINE INDENT return n NEW_LINE DEDENT return - 1 NEW_LINE
n = 12 NEW_LINE k = 3 NEW_LINE print ( kPrimeFactor ( n , k ) ) NEW_LINE n = 14 NEW_LINE k = 3 NEW_LINE print ( kPrimeFactor ( n , k ) ) NEW_LINE
MAX = 10001 NEW_LINE
def sieveOfEratosthenes ( s ) : NEW_LINE
prime = [ False for i in range ( MAX + 1 ) ] NEW_LINE
for i in range ( 2 , MAX + 1 , 2 ) : NEW_LINE INDENT s [ i ] = 2 ; NEW_LINE DEDENT
for i in range ( 3 , MAX , 2 ) : NEW_LINE INDENT if ( prime [ i ] == False ) : NEW_LINE DEDENT
s [ i ] = i NEW_LINE
for j in range ( i , MAX + 1 , 2 ) : NEW_LINE INDENT if j * j > MAX : NEW_LINE INDENT break NEW_LINE DEDENT if ( prime [ i * j ] == False ) : NEW_LINE INDENT prime [ i * j ] = True NEW_LINE DEDENT DEDENT
s [ i * j ] = i NEW_LINE
def kPrimeFactor ( n , k , s ) : NEW_LINE
while ( n > 1 ) : NEW_LINE INDENT if ( k == 1 ) : NEW_LINE INDENT return s [ n ] NEW_LINE DEDENT DEDENT
k -= 1 NEW_LINE
n //= s [ n ] NEW_LINE return - 1 NEW_LINE
s = [ - 1 for i in range ( MAX + 1 ) ] NEW_LINE sieveOfEratosthenes ( s ) NEW_LINE n = 12 NEW_LINE k = 3 NEW_LINE print ( kPrimeFactor ( n , k , s ) ) NEW_LINE n = 14 NEW_LINE k = 3 NEW_LINE print ( kPrimeFactor ( n , k , s ) ) NEW_LINE
import math as mt NEW_LINE
def sumDivisorsOfDivisors ( n ) : NEW_LINE
mp = dict ( ) NEW_LINE for j in range ( 2 , mt . ceil ( mt . sqrt ( n ) ) ) : NEW_LINE INDENT count = 0 NEW_LINE while ( n % j == 0 ) : NEW_LINE INDENT n //= j NEW_LINE count += 1 NEW_LINE DEDENT if ( count ) : NEW_LINE INDENT mp [ j ] = count NEW_LINE DEDENT DEDENT
if ( n != 1 ) : NEW_LINE INDENT mp [ n ] = 1 NEW_LINE DEDENT
ans = 1 NEW_LINE for it in mp : NEW_LINE INDENT pw = 1 NEW_LINE summ = 0 NEW_LINE for i in range ( mp [ it ] + 1 , 0 , - 1 ) : NEW_LINE INDENT summ += ( i * pw ) NEW_LINE pw *= it NEW_LINE DEDENT ans *= summ NEW_LINE DEDENT return ans NEW_LINE
n = 10 NEW_LINE print ( sumDivisorsOfDivisors ( n ) ) NEW_LINE
def fractionToDecimal ( numr , denr ) : NEW_LINE
res = " " NEW_LINE
mp = { } NEW_LINE
rem = numr % denr NEW_LINE
while ( ( rem != 0 ) and ( rem not in mp ) ) : NEW_LINE
mp [ rem ] = len ( res ) NEW_LINE
rem = rem * 10 NEW_LINE
res_part = rem // denr NEW_LINE res += str ( res_part ) NEW_LINE
rem = rem % denr NEW_LINE if ( rem == 0 ) : NEW_LINE return " " NEW_LINE else : NEW_LINE return res [ mp [ rem ] : ] NEW_LINE
numr , denr = 50 , 22 NEW_LINE res = fractionToDecimal ( numr , denr ) NEW_LINE if ( res == " " ) : NEW_LINE INDENT print ( " No ▁ recurring ▁ sequence " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Recurring ▁ sequence ▁ is " , res ) NEW_LINE DEDENT
def has0 ( x ) : NEW_LINE
while ( x != 0 ) : NEW_LINE
if ( x % 10 == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT x = x // 10 NEW_LINE return 0 NEW_LINE
def getCount ( n ) : NEW_LINE
count = 0 NEW_LINE
for i in range ( 1 , n + 1 ) : NEW_LINE INDENT count = count + has0 ( i ) NEW_LINE DEDENT return count NEW_LINE
n = 107 NEW_LINE print ( " Count ▁ of ▁ numbers ▁ from ▁ 1" , " ▁ to ▁ " , n , " ▁ is ▁ " , getCount ( n ) ) NEW_LINE
def squareRootExists ( n , p ) : NEW_LINE INDENT n = n % p NEW_LINE DEDENT
for x in range ( 2 , p , 1 ) : NEW_LINE INDENT if ( ( x * x ) % p == n ) : NEW_LINE INDENT return True NEW_LINE DEDENT DEDENT return False NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT p = 7 NEW_LINE n = 2 NEW_LINE if ( squareRootExists ( n , p ) == True ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT DEDENT
def largestPower ( n , p ) : NEW_LINE
x = 0 NEW_LINE
while n : NEW_LINE INDENT n /= p NEW_LINE x += n NEW_LINE DEDENT return x NEW_LINE
n = 10 ; p = 3 NEW_LINE print ( " The largest power of % d that divides % d ! is % d " % ( p , n , largestPower ( n , p ) ) ) NEW_LINE
def factorial ( n ) : NEW_LINE
return 1 if ( n == 1 or n == 0 ) else n * factorial ( n - 1 ) NEW_LINE
num = 5 NEW_LINE print ( " Factorial ▁ of " , num , " is " , factorial ( num ) ) NEW_LINE
def getBit ( num , i ) : NEW_LINE
return ( ( num & ( 1 << i ) ) != 0 ) NEW_LINE
def clearBit ( num , i ) : NEW_LINE
mask = ~ ( 1 << i ) NEW_LINE
return num & mask NEW_LINE
def Bitwise_AND_sum_i ( arr1 , arr2 , M , N ) : NEW_LINE
frequency = [ 0 ] * 32 NEW_LINE
for i in range ( N ) : NEW_LINE
bit_position = 0 NEW_LINE num = arr1 [ i ] NEW_LINE
while ( num ) : NEW_LINE
if ( num & 1 ) : NEW_LINE
frequency [ bit_position ] += 1 NEW_LINE
bit_position += 1 NEW_LINE
num >>= 1 NEW_LINE
for i in range ( M ) : NEW_LINE INDENT num = arr2 [ i ] NEW_LINE DEDENT
value_at_that_bit = 1 NEW_LINE
bitwise_AND_sum = 0 NEW_LINE
for bit_position in range ( 32 ) : NEW_LINE
if ( num & 1 ) : NEW_LINE
bitwise_AND_sum += frequency [ bit_position ] * value_at_that_bit NEW_LINE
num >>= 1 NEW_LINE
value_at_that_bit <<= 1 NEW_LINE
print ( bitwise_AND_sum , end = " ▁ " ) NEW_LINE return NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
arr1 = [ 1 , 2 , 3 ] NEW_LINE
arr2 = [ 1 , 2 , 3 ] NEW_LINE
N = len ( arr1 ) NEW_LINE
M = len ( arr2 ) NEW_LINE
Bitwise_AND_sum_i ( arr1 , arr2 , M , N ) NEW_LINE
def FlipBits ( n ) : NEW_LINE INDENT for bit in range ( 32 ) : NEW_LINE DEDENT
if ( ( n >> bit ) & 1 ) : NEW_LINE
n = n ^ ( 1 << bit ) NEW_LINE break NEW_LINE print ( " The ▁ number ▁ after ▁ unsetting ▁ the " , end = " ▁ " ) NEW_LINE print ( " rightmost ▁ set ▁ bit " , n ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 12 ; NEW_LINE FlipBits ( N ) NEW_LINE DEDENT
def bitwiseAndOdd ( n ) : NEW_LINE
result = 1 ; NEW_LINE
for i in range ( 3 , n + 1 , 2 ) : NEW_LINE INDENT result = ( result & i ) ; NEW_LINE DEDENT return result ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 10 ; NEW_LINE print ( bitwiseAndOdd ( n ) ) ; NEW_LINE DEDENT
def bitwiseAndOdd ( n ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT
n = 10 NEW_LINE print ( bitwiseAndOdd ( n ) ) NEW_LINE
def reverseBits ( n ) : NEW_LINE INDENT rev = 0 NEW_LINE DEDENT
while ( n > 0 ) : NEW_LINE
rev = rev << 1 NEW_LINE
' NEW_LINE INDENT if ( n & 1 == 1 ) : NEW_LINE INDENT rev = rev ^ 1 NEW_LINE DEDENT DEDENT
n = n >> 1 NEW_LINE
return rev NEW_LINE
n = 11 NEW_LINE print ( reverseBits ( n ) ) NEW_LINE
def countgroup ( a , n ) : NEW_LINE INDENT xs = 0 NEW_LINE for i in range ( n ) : NEW_LINE INDENT xs = xs ^ a [ i ] NEW_LINE DEDENT DEDENT
if xs == 0 : NEW_LINE INDENT return ( 1 << ( n - 1 ) ) - 1 NEW_LINE DEDENT return 0 NEW_LINE
a = [ 1 , 2 , 3 ] NEW_LINE n = len ( a ) NEW_LINE print ( countgroup ( a , n ) ) NEW_LINE
def bitExtracted ( number , k , p ) : NEW_LINE INDENT return ( ( ( 1 << k ) - 1 ) & ( number >> ( p - 1 ) ) ) ; NEW_LINE DEDENT
number = 171 NEW_LINE k = 5 NEW_LINE p = 2 NEW_LINE print " The ▁ extracted ▁ number ▁ is ▁ " , bitExtracted ( number , k , p ) NEW_LINE
def findMax ( num ) : NEW_LINE INDENT num_copy = num NEW_LINE DEDENT
j = 4 * 8 - 1 ; NEW_LINE i = 0 NEW_LINE while ( i < j ) : NEW_LINE
m = ( num_copy >> i ) & 1 NEW_LINE n = ( num_copy >> j ) & 1 NEW_LINE
if ( m > n ) : NEW_LINE INDENT x = ( 1 << i 1 << j ) NEW_LINE num = num ^ x NEW_LINE DEDENT i += 1 NEW_LINE j -= 1 NEW_LINE return num NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT num = 4 NEW_LINE print ( findMax ( num ) ) NEW_LINE DEDENT
def isAMultipleOf4 ( n ) : NEW_LINE
if ( ( n & 3 ) == 0 ) : NEW_LINE INDENT return " Yes " NEW_LINE DEDENT
return " No " NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 16 NEW_LINE print ( isAMultipleOf4 ( n ) ) NEW_LINE DEDENT
def square ( n ) : NEW_LINE
if ( n < 0 ) : NEW_LINE INDENT n = - n NEW_LINE DEDENT
res = n NEW_LINE
for i in range ( 1 , n ) : NEW_LINE INDENT res += n NEW_LINE DEDENT return res NEW_LINE
for n in range ( 1 , 6 ) : NEW_LINE INDENT print ( " n ▁ = " , n , end = " , ▁ " ) NEW_LINE print ( " n ^ 2 ▁ = " , square ( n ) ) NEW_LINE DEDENT
def PointInKSquares ( n , a , k ) : NEW_LINE INDENT a . sort ( ) NEW_LINE return a [ n - k ] NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT k = 2 NEW_LINE a = [ 1 , 2 , 3 , 4 ] NEW_LINE n = len ( a ) NEW_LINE x = PointInKSquares ( n , a , k ) NEW_LINE print ( " ( " , x , " , " , x , " ) " ) NEW_LINE DEDENT
def answer ( n ) : NEW_LINE
dp = [ 0 ] * 10 NEW_LINE
prev = [ 0 ] * 10 NEW_LINE
if ( n == 1 ) : NEW_LINE INDENT return 10 NEW_LINE DEDENT
for j in range ( 0 , 10 ) : NEW_LINE INDENT dp [ j ] = 1 NEW_LINE DEDENT
for i in range ( 2 , n + 1 ) : NEW_LINE INDENT for j in range ( 0 , 10 ) : NEW_LINE INDENT prev [ j ] = dp [ j ] NEW_LINE DEDENT for j in range ( 0 , 10 ) : NEW_LINE DEDENT
if ( j == 0 ) : NEW_LINE INDENT dp [ j ] = prev [ j + 1 ] NEW_LINE DEDENT
elif ( j == 9 ) : NEW_LINE INDENT dp [ j ] = prev [ j - 1 ] NEW_LINE DEDENT
else : NEW_LINE INDENT dp [ j ] = prev [ j - 1 ] + prev [ j + 1 ] NEW_LINE DEDENT
sum = 0 NEW_LINE for j in range ( 1 , 10 ) : NEW_LINE INDENT sum = sum + dp [ j ] NEW_LINE DEDENT return sum NEW_LINE
n = 2 NEW_LINE print ( answer ( n ) ) NEW_LINE
MAX = 100000 ; NEW_LINE
catalan = [ 0 ] * MAX ; NEW_LINE
def catalanDP ( n ) : NEW_LINE
catalan [ 0 ] = catalan [ 1 ] = 1 ; NEW_LINE
for i in range ( 2 , n + 1 ) : NEW_LINE INDENT catalan [ i ] = 0 ; NEW_LINE for j in range ( i ) : NEW_LINE INDENT catalan [ i ] += ( catalan [ j ] * catalan [ i - j - 1 ] ) ; NEW_LINE DEDENT DEDENT
def CatalanSequence ( arr , n ) : NEW_LINE
catalanDP ( n ) ; NEW_LINE s = set ( ) ; NEW_LINE
a = 1 ; b = 1 ; NEW_LINE
s . add ( a ) ; NEW_LINE if ( n >= 2 ) : NEW_LINE INDENT s . add ( b ) ; NEW_LINE DEDENT for i in range ( 2 , n ) : NEW_LINE INDENT s . add ( catalan [ i ] ) ; NEW_LINE DEDENT temp = set ( ) NEW_LINE for i in range ( n ) : NEW_LINE
if arr [ i ] in s : NEW_LINE INDENT temp . add ( arr [ i ] ) NEW_LINE DEDENT s = s - temp ; NEW_LINE
return len ( s ) ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 1 , 1 , 2 , 5 , 41 ] ; NEW_LINE n = len ( arr ) NEW_LINE print ( CatalanSequence ( arr , n ) ) ; NEW_LINE DEDENT
def odd_indices ( arr ) : NEW_LINE INDENT sum = 0 NEW_LINE DEDENT
for k in range ( 0 , len ( arr ) , 2 ) : NEW_LINE INDENT check = composite ( arr [ k ] ) NEW_LINE DEDENT
sum += arr [ k ] if check == 1 else 0 NEW_LINE
print ( sum ) NEW_LINE
def composite ( n ) : NEW_LINE INDENT flag = 0 NEW_LINE c = 0 NEW_LINE DEDENT
for j in range ( 1 , n + 1 ) : NEW_LINE INDENT if ( n % j == 0 ) : NEW_LINE INDENT c += 1 NEW_LINE DEDENT DEDENT
if ( c >= 3 ) : NEW_LINE INDENT flag = 1 NEW_LINE DEDENT return flag NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 13 , 5 , 8 , 16 , 25 ] NEW_LINE odd_indices ( arr ) NEW_LINE DEDENT
import math NEW_LINE
def preprocess ( p , x , y , n ) : NEW_LINE INDENT for i in range ( n ) : NEW_LINE INDENT p [ i ] = x [ i ] * x [ i ] + y [ i ] * y [ i ] NEW_LINE DEDENT p . sort ( ) NEW_LINE DEDENT
def query ( p , n , rad ) : NEW_LINE INDENT start = 0 NEW_LINE end = n - 1 NEW_LINE while ( ( end - start ) > 1 ) : NEW_LINE INDENT mid = ( start + end ) // 2 NEW_LINE tp = math . sqrt ( p [ mid ] ) NEW_LINE if ( tp > ( rad * 1.0 ) ) : NEW_LINE INDENT end = mid - 1 NEW_LINE DEDENT else : NEW_LINE INDENT start = mid NEW_LINE DEDENT DEDENT tp1 = math . sqrt ( p [ start ] ) NEW_LINE tp2 = math . sqrt ( p [ end ] ) NEW_LINE if ( tp1 > ( rad * 1.0 ) ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT elif ( tp2 <= ( rad * 1.0 ) ) : NEW_LINE INDENT return end + 1 NEW_LINE DEDENT else : NEW_LINE INDENT return start + 1 NEW_LINE DEDENT DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT x = [ 1 , 2 , 3 , - 1 , 4 ] NEW_LINE y = [ 1 , 2 , 3 , - 1 , 4 ] NEW_LINE n = len ( x ) NEW_LINE DEDENT
p = [ 0 ] * n NEW_LINE preprocess ( p , x , y , n ) NEW_LINE
print ( query ( p , n , 3 ) ) NEW_LINE
print ( query ( p , n , 32 ) ) NEW_LINE
def count ( N ) : NEW_LINE
odd_indices = N // 2 NEW_LINE
even_indices = N // 2 + N % 2 NEW_LINE
arrange_odd = 4 ** odd_indices NEW_LINE
arrange_even = 5 ** even_indices NEW_LINE
return arrange_odd * arrange_even NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT N = 4 NEW_LINE DEDENT
print ( count ( N ) ) NEW_LINE
def isSpiralSorted ( arr , n ) : NEW_LINE
start = 0 ; NEW_LINE
end = n - 1 ; NEW_LINE while ( start < end ) : NEW_LINE
if ( arr [ start ] > arr [ end ] ) : NEW_LINE INDENT return False ; NEW_LINE DEDENT
start += 1 ; NEW_LINE
if ( arr [ end ] > arr [ start ] ) : NEW_LINE INDENT return False ; NEW_LINE DEDENT
end -= 1 ; NEW_LINE return True ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 1 , 10 , 14 , 20 , 18 , 12 , 5 ] ; NEW_LINE N = len ( arr ) ; NEW_LINE DEDENT
if ( isSpiralSorted ( arr , N ) ) : NEW_LINE INDENT print ( " YES " ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NO " ) ; NEW_LINE DEDENT
def findWordsSameRow ( arr ) : NEW_LINE
mp = { ' q ' : 1 , ' w ' : 1 , ' e ' : 1 , ' r ' : 1 , ' t ' : 1 , ' y ' : 1 , ' u ' : 1 , ' o ' : 1 , ' p ' : 1 , ' i ' : 1 , ' a ' : 2 , ' s ' : 2 , ' d ' : 2 , ' f ' : 2 , ' g ' : 2 , ' h ' : 2 , ' j ' : 2 , ' k ' : 2 , ' l ' : 2 , ' z ' : 3 , ' x ' : 3 , ' c ' : 3 , ' v ' : 3 , ' b ' : 3 , ' n ' : 3 , ' m ' : 3 } NEW_LINE
for word in arr : NEW_LINE
if ( len ( word ) != 0 ) : NEW_LINE
flag = True NEW_LINE
rowNum = mp [ word [ 0 ] . lower ( ) ] NEW_LINE
M = len ( word ) NEW_LINE
for i in range ( 1 , M ) : NEW_LINE
if ( mp [ word [ i ] . lower ( ) ] != rowNum ) : NEW_LINE
flag = False NEW_LINE break NEW_LINE
if ( flag ) : NEW_LINE
print ( word , end = ' ▁ ' ) NEW_LINE
words = [ " Yeti " , " Had " , " GFG " , " comment " ] NEW_LINE findWordsSameRow ( words ) NEW_LINE
maxN = 2002 NEW_LINE
def countSubsequece ( a , n ) : NEW_LINE
answer = 0 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT for j in range ( i + 1 , n ) : NEW_LINE INDENT for k in range ( j + 1 , n ) : NEW_LINE INDENT for l in range ( k + 1 , n ) : NEW_LINE DEDENT DEDENT DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT a = [ 1 , 2 , 3 , 2 , 1 , 3 , 2 ] NEW_LINE print ( countSubsequece ( a , 7 ) ) NEW_LINE DEDENT
import sys NEW_LINE
def minDistChar ( s ) : NEW_LINE INDENT n = len ( s ) NEW_LINE DEDENT
first = [ ] NEW_LINE last = [ ] NEW_LINE
for i in range ( 26 ) : NEW_LINE INDENT first . append ( - 1 ) NEW_LINE last . append ( - 1 ) NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE
if ( first [ ord ( s [ i ] ) - ord ( ' a ' ) ] == - 1 ) : NEW_LINE INDENT first [ ord ( s [ i ] ) - ord ( ' a ' ) ] = i NEW_LINE DEDENT
last [ ord ( s [ i ] ) - ord ( ' a ' ) ] = i NEW_LINE
min = sys . maxsize NEW_LINE ans = '1' NEW_LINE
for i in range ( 26 ) : NEW_LINE
if ( last [ i ] == first [ i ] ) : NEW_LINE INDENT continue NEW_LINE DEDENT
if ( min > last [ i ] - first [ i ] ) : NEW_LINE INDENT min = last [ i ] - first [ i ] NEW_LINE ans = i + ord ( ' a ' ) NEW_LINE DEDENT
return chr ( ans ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT str = " geeksforgeeks " NEW_LINE DEDENT
print ( minDistChar ( str ) ) NEW_LINE
n = 3 NEW_LINE
def minSteps ( arr ) : NEW_LINE
v = [ [ 0 for i in range ( n ) ] for j in range ( n ) ] NEW_LINE
q = [ [ 0 , 0 ] ] NEW_LINE
depth = 0 NEW_LINE
while ( len ( q ) != 0 ) : NEW_LINE
x = len ( q ) NEW_LINE while ( x > 0 ) : NEW_LINE
y = q [ 0 ] NEW_LINE
i = y [ 0 ] NEW_LINE j = y [ 1 ] NEW_LINE q . remove ( q [ 0 ] ) NEW_LINE x -= 1 NEW_LINE
if ( v [ i ] [ j ] ) : NEW_LINE INDENT continue NEW_LINE DEDENT
if ( i == n - 1 and j == n - 1 ) : NEW_LINE INDENT return depth NEW_LINE DEDENT
v [ i ] [ j ] = 1 NEW_LINE
if ( i + arr [ i ] [ j ] < n ) : NEW_LINE INDENT q . append ( [ i + arr [ i ] [ j ] , j ] ) NEW_LINE DEDENT if ( j + arr [ i ] [ j ] < n ) : NEW_LINE INDENT q . append ( [ i , j + arr [ i ] [ j ] ] ) NEW_LINE DEDENT depth += 1 NEW_LINE return - 1 NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ [ 1 , 1 , 1 ] , [ 1 , 1 , 1 ] , [ 1 , 1 , 1 ] ] NEW_LINE print ( minSteps ( arr ) ) NEW_LINE DEDENT
import sys NEW_LINE
def solve ( a , n ) : NEW_LINE INDENT max1 = - sys . maxsize - 1 NEW_LINE for i in range ( 0 , n , 1 ) : NEW_LINE INDENT for j in range ( 0 , n , 1 ) : NEW_LINE INDENT if ( abs ( a [ i ] - a [ j ] ) > max1 ) : NEW_LINE INDENT max1 = abs ( a [ i ] - a [ j ] ) NEW_LINE DEDENT DEDENT DEDENT return max1 NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ - 1 , 2 , 3 , - 4 , - 10 , 22 ] NEW_LINE size = len ( arr ) NEW_LINE print ( " Largest ▁ gap ▁ is ▁ : " , solve ( arr , size ) ) NEW_LINE DEDENT
def solve ( a , n ) : NEW_LINE INDENT min1 = a [ 0 ] NEW_LINE max1 = a [ 0 ] NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT if ( a [ i ] > max1 ) : NEW_LINE INDENT max1 = a [ i ] NEW_LINE DEDENT if ( a [ i ] < min1 ) : NEW_LINE INDENT min1 = a [ i ] NEW_LINE DEDENT DEDENT return abs ( min1 - max1 ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ - 1 , 2 , 3 , 4 , - 10 ] NEW_LINE size = len ( arr ) NEW_LINE print ( " Largest ▁ gap ▁ is ▁ : ▁ " , solve ( arr , size ) ) NEW_LINE DEDENT
def replaceOriginal ( s , n ) : NEW_LINE
r = [ ' ▁ ' ] * n NEW_LINE
for i in range ( n ) : NEW_LINE
r [ i ] = s [ n - 1 - i ] NEW_LINE
if ( s [ i ] != ' a ' and s [ i ] != ' e ' and s [ i ] != ' i ' and s [ i ] != ' o ' and s [ i ] != ' u ' ) : NEW_LINE INDENT print ( r [ i ] , end = " " ) NEW_LINE DEDENT print ( ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT s = " geeksforgeeks " NEW_LINE n = len ( s ) NEW_LINE replaceOriginal ( s , n ) NEW_LINE DEDENT
def sameStrings ( str1 , str2 ) : NEW_LINE INDENT N = len ( str1 ) NEW_LINE M = len ( str2 ) NEW_LINE DEDENT
if ( N != M ) : NEW_LINE INDENT return False NEW_LINE DEDENT
a , b = [ 0 ] * 256 , [ 0 ] * 256 NEW_LINE
for i in range ( N ) : NEW_LINE INDENT a [ ord ( str1 [ i ] ) - ord ( ' a ' ) ] += 1 NEW_LINE b [ ord ( str2 [ i ] ) - ord ( ' a ' ) ] += 1 NEW_LINE DEDENT
i = 0 NEW_LINE while ( i < 256 ) : NEW_LINE INDENT if ( ( a [ i ] == 0 and b [ i ] == 0 ) or ( a [ i ] != 0 and b [ i ] != 0 ) ) : NEW_LINE INDENT i += 1 NEW_LINE DEDENT DEDENT
else : NEW_LINE INDENT return False NEW_LINE DEDENT
a = sorted ( a ) NEW_LINE b = sorted ( b ) NEW_LINE
for i in range ( 256 ) : NEW_LINE
if ( a [ i ] != b [ i ] ) : NEW_LINE INDENT return False NEW_LINE DEDENT
return True NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT S1 , S2 = " cabbba " , " abbccc " NEW_LINE if ( sameStrings ( S1 , S2 ) ) : NEW_LINE INDENT print ( " YES " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NO " ) NEW_LINE DEDENT DEDENT
def solution ( A , B , C ) : NEW_LINE INDENT arr = [ 0 ] * 3 NEW_LINE DEDENT
arr [ 0 ] = A NEW_LINE arr [ 1 ] = B NEW_LINE arr [ 2 ] = C NEW_LINE
arr = sorted ( arr ) NEW_LINE
if ( arr [ 2 ] < arr [ 0 ] + arr [ 1 ] ) : NEW_LINE INDENT return ( ( arr [ 0 ] + arr [ 1 ] + arr [ 2 ] ) // 2 ) NEW_LINE DEDENT
else : NEW_LINE INDENT return ( arr [ 0 ] + arr [ 1 ] ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE
A = 8 NEW_LINE B = 1 NEW_LINE C = 5 NEW_LINE
print ( solution ( A , B , C ) ) NEW_LINE
def search ( arr , l , h , key ) : NEW_LINE INDENT if ( l > h ) : NEW_LINE INDENT return - 1 ; NEW_LINE DEDENT mid = ( l + h ) // 2 ; NEW_LINE if ( arr [ mid ] == key ) : NEW_LINE INDENT return mid ; NEW_LINE DEDENT DEDENT
if ( ( arr [ l ] == arr [ mid ] ) and ( arr [ h ] == arr [ mid ] ) ) : NEW_LINE INDENT l += 1 ; NEW_LINE h -= 1 ; NEW_LINE return search ( arr , l , h , key ) NEW_LINE DEDENT
if ( arr [ l ] <= arr [ mid ] ) : NEW_LINE
if ( key >= arr [ l ] and key <= arr [ mid ] ) : NEW_LINE INDENT return search ( arr , l , mid - 1 , key ) ; NEW_LINE DEDENT
return search ( arr , mid + 1 , h , key ) ; NEW_LINE
if ( key >= arr [ mid ] and key <= arr [ h ] ) : NEW_LINE INDENT return search ( arr , mid + 1 , h , key ) ; NEW_LINE DEDENT return search ( arr , l , mid - 1 , key ) ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 3 , 3 , 1 , 2 , 3 , 3 ] ; NEW_LINE n = len ( arr ) ; NEW_LINE key = 3 ; NEW_LINE print ( search ( arr , 0 , n - 1 , key ) ) ; NEW_LINE DEDENT
def getSortedString ( s , n ) : NEW_LINE
v1 = [ ] NEW_LINE v2 = [ ] NEW_LINE for i in range ( n ) : NEW_LINE INDENT if ( s [ i ] >= ' a ' and s [ i ] <= ' z ' ) : NEW_LINE INDENT v1 . append ( s [ i ] ) NEW_LINE DEDENT if ( s [ i ] >= ' A ' and s [ i ] <= ' Z ' ) : NEW_LINE INDENT v2 . append ( s [ i ] ) NEW_LINE DEDENT DEDENT
v1 = sorted ( v1 ) NEW_LINE v2 = sorted ( v2 ) NEW_LINE i = 0 NEW_LINE j = 0 NEW_LINE for k in range ( n ) : NEW_LINE
if ( s [ k ] >= ' a ' and s [ k ] <= ' z ' ) : NEW_LINE INDENT s [ k ] = v1 [ i ] NEW_LINE i += 1 NEW_LINE DEDENT
elif ( s [ k ] >= ' A ' and s [ k ] <= ' Z ' ) : NEW_LINE INDENT s [ k ] = v2 [ j ] NEW_LINE j += 1 NEW_LINE DEDENT
return " " . join ( s ) NEW_LINE
s = " gEeksfOrgEEkS " NEW_LINE ss = [ i for i in s ] NEW_LINE n = len ( ss ) NEW_LINE print ( getSortedString ( ss , n ) ) NEW_LINE
def check ( s ) : NEW_LINE
l = len ( s ) NEW_LINE
s = ' ' . join ( sorted ( s ) ) NEW_LINE
for i in range ( 1 , l ) : NEW_LINE
if ord ( s [ i ] ) - ord ( s [ i - 1 ] ) != 1 : NEW_LINE INDENT return False NEW_LINE DEDENT return True NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
string = " dcef " NEW_LINE if check ( string ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
string = " xyza " NEW_LINE if check ( string ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
def minElements ( arr , n ) : NEW_LINE
halfSum = 0 NEW_LINE for i in range ( n ) : NEW_LINE INDENT halfSum = halfSum + arr [ i ] NEW_LINE DEDENT halfSum = int ( halfSum / 2 ) NEW_LINE
arr . sort ( reverse = True ) NEW_LINE res = 0 NEW_LINE curr_sum = 0 NEW_LINE for i in range ( n ) : NEW_LINE INDENT curr_sum += arr [ i ] NEW_LINE res += 1 NEW_LINE DEDENT
if curr_sum > halfSum : NEW_LINE INDENT return res NEW_LINE DEDENT return res NEW_LINE
arr = [ 3 , 1 , 7 , 1 ] NEW_LINE n = len ( arr ) NEW_LINE print ( minElements ( arr , n ) ) NEW_LINE
def arrayElementEqual ( arr , N ) : NEW_LINE
sum = 0 NEW_LINE
for i in range ( N ) : NEW_LINE INDENT sum += arr [ i ] NEW_LINE DEDENT
if ( sum % N == 0 ) : NEW_LINE INDENT print ( ' Yes ' ) NEW_LINE DEDENT
else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
arr = [ 1 , 5 , 6 , 4 ] NEW_LINE
N = len ( arr ) NEW_LINE arrayElementEqual ( arr , N ) NEW_LINE
def findMaxValByRearrArr ( arr , N ) : NEW_LINE
res = 0 ; NEW_LINE
res = ( N * ( N + 1 ) ) // 2 ; NEW_LINE return res ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 3 , 2 , 1 ] ; NEW_LINE N = len ( arr ) ; NEW_LINE print ( findMaxValByRearrArr ( arr , N ) ) ; NEW_LINE DEDENT
def MaximumSides ( n ) : NEW_LINE
if ( n < 4 ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
if n % 2 == 0 : NEW_LINE INDENT return n // 2 NEW_LINE DEDENT return - 1 NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
N = 8 NEW_LINE
print ( MaximumSides ( N ) ) NEW_LINE
def pairProductMean ( arr , N ) : NEW_LINE
suffixSumArray = [ 0 ] * N NEW_LINE suffixSumArray [ N - 1 ] = arr [ N - 1 ] NEW_LINE
for i in range ( N - 2 , - 1 , - 1 ) : NEW_LINE INDENT suffixSumArray [ i ] = suffixSumArray [ i + 1 ] + arr [ i ] NEW_LINE DEDENT
length = ( N * ( N - 1 ) ) // 2 NEW_LINE
res = 0 NEW_LINE for i in range ( N - 1 ) : NEW_LINE INDENT res += arr [ i ] * suffixSumArray [ i + 1 ] NEW_LINE DEDENT
mean = 0 NEW_LINE
if ( length != 0 ) : NEW_LINE INDENT mean = res / length NEW_LINE DEDENT else : NEW_LINE INDENT mean = 0 NEW_LINE DEDENT
return mean NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
arr = [ 1 , 2 , 4 , 8 ] NEW_LINE N = len ( arr ) NEW_LINE
print ( round ( pairProductMean ( arr , N ) , 2 ) ) NEW_LINE
def ncr ( n , k ) : NEW_LINE INDENT res = 1 NEW_LINE DEDENT
if ( k > n - k ) : NEW_LINE INDENT k = n - k NEW_LINE DEDENT
for i in range ( k ) : NEW_LINE INDENT res *= ( n - i ) NEW_LINE res //= ( i + 1 ) NEW_LINE DEDENT return res NEW_LINE
def countPath ( N , M , K ) : NEW_LINE INDENT answer = 0 NEW_LINE if ( K >= 2 ) : NEW_LINE INDENT answer = 0 NEW_LINE DEDENT elif ( K == 0 ) : NEW_LINE INDENT answer = ncr ( N + M - 2 , N - 1 ) NEW_LINE DEDENT else : NEW_LINE DEDENT
answer = ncr ( N + M - 2 , N - 1 ) NEW_LINE
X = ( N - 1 ) // 2 + ( M - 1 ) // 2 NEW_LINE Y = ( N - 1 ) // 2 NEW_LINE midCount = ncr ( X , Y ) NEW_LINE
X = ( ( N - 1 ) - ( N - 1 ) // 2 ) + NEW_LINE INDENT ( ( M - 1 ) - ( M - 1 ) // 2 ) NEW_LINE DEDENT Y = ( ( N - 1 ) - ( N - 1 ) // 2 ) NEW_LINE midCount *= ncr ( X , Y ) NEW_LINE answer -= midCount NEW_LINE return answer NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 3 NEW_LINE M = 3 NEW_LINE K = 1 NEW_LINE print ( countPath ( N , M , K ) ) NEW_LINE DEDENT
def find_max ( v , n ) : NEW_LINE
count = 0 NEW_LINE if ( n >= 2 ) : NEW_LINE INDENT count = 2 NEW_LINE DEDENT else : NEW_LINE INDENT count = 1 NEW_LINE DEDENT
for i in range ( 1 , n - 1 ) : NEW_LINE
if ( v [ i - 1 ] [ 0 ] > ( v [ i ] [ 0 ] + v [ i ] [ 1 ] ) ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT
elif ( v [ i + 1 ] [ 0 ] > ( v [ i ] [ 0 ] + v [ i ] [ 1 ] ) ) : NEW_LINE INDENT count += 1 NEW_LINE v [ i ] [ 0 ] = v [ i ] [ 0 ] + v [ i ] [ 1 ] NEW_LINE DEDENT
else : NEW_LINE INDENT continue NEW_LINE DEDENT
return count NEW_LINE
n = 3 NEW_LINE v = [ ] NEW_LINE v . append ( [ 10 , 20 ] ) NEW_LINE v . append ( [ 15 , 10 ] ) NEW_LINE v . append ( [ 20 , 16 ] ) NEW_LINE print ( find_max ( v , n ) ) NEW_LINE
def numberofsubstrings ( str , k , charArray ) : NEW_LINE INDENT N = len ( str ) NEW_LINE DEDENT
available = [ 0 ] * 26 NEW_LINE
for i in range ( 0 , k ) : NEW_LINE INDENT available [ ord ( charArray [ i ] ) - ord ( ' a ' ) ] = 1 NEW_LINE DEDENT
lastPos = - 1 NEW_LINE
ans = ( N * ( N + 1 ) ) / 2 NEW_LINE
for i in range ( 0 , N ) : NEW_LINE
if ( available [ ord ( str [ i ] ) - ord ( ' a ' ) ] == 0 ) : NEW_LINE
ans -= ( ( i - lastPos ) * ( N - i ) ) NEW_LINE
lastPos = i NEW_LINE
print ( int ( ans ) ) NEW_LINE
str = " abcb " NEW_LINE k = 2 NEW_LINE
charArray = [ ' a ' , ' b ' ] NEW_LINE
numberofsubstrings ( str , k , charArray ) NEW_LINE
def minCost ( N , P , Q ) : NEW_LINE
cost = 0 NEW_LINE
while ( N > 0 ) : NEW_LINE INDENT if ( N & 1 ) : NEW_LINE INDENT cost += P NEW_LINE N -= 1 NEW_LINE DEDENT else : NEW_LINE INDENT temp = N // 2 ; NEW_LINE DEDENT DEDENT
if ( temp * P > Q ) : NEW_LINE INDENT cost += Q NEW_LINE DEDENT
else : NEW_LINE INDENT cost += P * temp NEW_LINE DEDENT N //= 2 NEW_LINE return cost NEW_LINE
N = 9 NEW_LINE P = 5 NEW_LINE Q = 1 NEW_LINE print ( minCost ( N , P , Q ) ) NEW_LINE
def numberOfWays ( n , k ) : NEW_LINE
dp = [ 0 for i in range ( 1000 ) ] NEW_LINE
dp [ 0 ] = 1 NEW_LINE
for i in range ( 1 , k + 1 , 1 ) : NEW_LINE
numWays = 0 NEW_LINE
for j in range ( n ) : NEW_LINE INDENT numWays += dp [ j ] NEW_LINE DEDENT
for j in range ( n ) : NEW_LINE INDENT dp [ j ] = numWays - dp [ j ] NEW_LINE DEDENT
print ( dp [ 0 ] ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
N = 5 NEW_LINE K = 3 NEW_LINE
numberOfWays ( N , K ) NEW_LINE
def findMinCost ( arr , X , n , i = 0 ) : NEW_LINE
if ( X <= 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT if ( i >= n ) : NEW_LINE INDENT return 10 ** 8 NEW_LINE DEDENT
inc = findMinCost ( arr , X - arr [ i ] [ 0 ] , n , i + 1 ) NEW_LINE if ( inc != 10 ** 8 ) : NEW_LINE INDENT inc += arr [ i ] [ 1 ] NEW_LINE DEDENT
exc = findMinCost ( arr , X , n , i + 1 ) NEW_LINE
return min ( inc , exc ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
arr = [ [ 4 , 3 ] , [ 3 , 2 ] , [ 2 , 4 ] , [ 1 , 3 ] , [ 4 , 2 ] ] NEW_LINE X = 7 NEW_LINE
n = len ( arr ) NEW_LINE ans = findMinCost ( arr , X , n ) NEW_LINE
if ( ans != 10 ** 8 ) : NEW_LINE INDENT print ( ans ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( - 1 ) NEW_LINE DEDENT
def find ( N , sum ) : NEW_LINE
if ( sum > 6 * N or sum < N ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT if ( N == 1 ) : NEW_LINE INDENT if ( sum >= 1 and sum <= 6 ) : NEW_LINE INDENT return 1.0 / 6 NEW_LINE DEDENT else : NEW_LINE INDENT return 0 NEW_LINE DEDENT DEDENT s = 0 NEW_LINE for i in range ( 1 , 7 ) : NEW_LINE INDENT s = s + find ( N - 1 , sum - i ) / 6 NEW_LINE DEDENT return s NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT N = 4 NEW_LINE a = 13 NEW_LINE b = 17 NEW_LINE probability = 0.0 NEW_LINE for sum in range ( a , b + 1 ) : NEW_LINE INDENT probability = probability + find ( N , sum ) NEW_LINE DEDENT DEDENT
print ( round ( probability , 6 ) ) NEW_LINE
def minDays ( n ) : NEW_LINE
if n < 1 : NEW_LINE INDENT return n NEW_LINE DEDENT
cnt = 1 + min ( n % 2 + minDays ( n // 2 ) , n % 3 + minDays ( n // 3 ) ) NEW_LINE
return cnt NEW_LINE
N = 6 NEW_LINE
print ( str ( minDays ( N ) ) ) NEW_LINE
