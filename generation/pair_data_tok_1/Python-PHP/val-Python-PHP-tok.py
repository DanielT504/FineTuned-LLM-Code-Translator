def Conversion ( centi ) : NEW_LINE INDENT pixels = ( 96 * centi ) / 2.54 NEW_LINE print ( round ( pixels , 2 ) ) NEW_LINE DEDENT
centi = 15 NEW_LINE Conversion ( centi ) NEW_LINE
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
def printKDistinct ( arr , n , k ) : NEW_LINE INDENT dist_count = 0 NEW_LINE for i in range ( n ) : NEW_LINE DEDENT
j = 0 NEW_LINE while j < n : NEW_LINE INDENT if ( i != j and arr [ j ] == arr [ i ] ) : NEW_LINE INDENT break NEW_LINE DEDENT j += 1 NEW_LINE DEDENT
if ( j == n ) : NEW_LINE INDENT dist_count += 1 NEW_LINE DEDENT if ( dist_count == k ) : NEW_LINE INDENT return arr [ i ] NEW_LINE DEDENT return - 1 NEW_LINE
ar = [ 1 , 2 , 1 , 3 , 4 , 2 ] NEW_LINE n = len ( ar ) NEW_LINE k = 2 NEW_LINE print ( printKDistinct ( ar , n , k ) ) NEW_LINE
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
if ( mid < high and arr [ mid + 1 ] < arr [ mid ] and mid > 0 and arr [ mid ] > arr [ mid - 1 ] ) : NEW_LINE INDENT return arr [ mid ] NEW_LINE DEDENT
if ( arr [ low ] > arr [ mid ] ) : NEW_LINE INDENT return findMax ( arr , low , mid - 1 ) NEW_LINE DEDENT else : NEW_LINE INDENT return findMax ( arr , mid + 1 , high ) NEW_LINE DEDENT
arr = [ 6 , 5 , 4 , 3 , 2 , 1 ] NEW_LINE n = len ( arr ) NEW_LINE print ( findMax ( arr , 0 , n - 1 ) ) NEW_LINE
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
def print2Smallest ( arr ) : NEW_LINE
arr_size = len ( arr ) NEW_LINE if arr_size < 2 : NEW_LINE INDENT print " Invalid ▁ Input " NEW_LINE return NEW_LINE DEDENT first = second = sys . maxint NEW_LINE for i in range ( 0 , arr_size ) : NEW_LINE
if arr [ i ] < first : NEW_LINE INDENT second = first NEW_LINE first = arr [ i ] NEW_LINE DEDENT
elif ( arr [ i ] < second and arr [ i ] != first ) : NEW_LINE INDENT second = arr [ i ] ; NEW_LINE DEDENT if ( second == sys . maxint ) : NEW_LINE print " No ▁ second ▁ smallest ▁ element " NEW_LINE else : NEW_LINE print ' The ▁ smallest ▁ element ▁ is ' , first , ' and ' ' ▁ second ▁ smallest ▁ element ▁ is ' , second NEW_LINE
arr = [ 12 , 13 , 1 , 10 , 34 , 1 ] NEW_LINE print2Smallest ( arr ) NEW_LINE
def isSubsetSum ( arr , n , sum ) : NEW_LINE
subset = [ [ False for j in range ( sum + 1 ) ] for i in range ( 3 ) ] NEW_LINE for i in range ( n + 1 ) : NEW_LINE INDENT for j in range ( sum + 1 ) : NEW_LINE DEDENT
if ( j == 0 ) : NEW_LINE INDENT subset [ i % 2 ] [ j ] = True NEW_LINE DEDENT
elif ( i == 0 ) : NEW_LINE INDENT subset [ i % 2 ] [ j ] = False NEW_LINE DEDENT elif ( arr [ i - 1 ] <= j ) : NEW_LINE INDENT subset [ i % 2 ] [ j ] = subset [ ( i + 1 ) % 2 ] [ j - arr [ i - 1 ] ] or subset [ ( i + 1 ) % 2 ] [ j ] NEW_LINE DEDENT else : NEW_LINE INDENT subset [ i % 2 ] [ j ] = subset [ ( i + 1 ) % 2 ] [ j ] NEW_LINE DEDENT return subset [ n % 2 ] [ sum ] NEW_LINE
arr = [ 6 , 2 , 5 ] NEW_LINE sum = 7 NEW_LINE n = len ( arr ) NEW_LINE if ( isSubsetSum ( arr , n , sum ) == True ) : NEW_LINE INDENT print ( " There ▁ exists ▁ a ▁ subset ▁ with ▁ given ▁ sum " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No ▁ subset ▁ exists ▁ with ▁ given ▁ sum " ) NEW_LINE DEDENT
def findMaxSum ( arr , n ) : NEW_LINE INDENT res = - sys . maxsize - 1 NEW_LINE for i in range ( n ) : NEW_LINE INDENT prefix_sum = arr [ i ] NEW_LINE for j in range ( i ) : NEW_LINE INDENT prefix_sum += arr [ j ] NEW_LINE DEDENT suffix_sum = arr [ i ] NEW_LINE j = n - 1 NEW_LINE while ( j > i ) : NEW_LINE INDENT suffix_sum += arr [ j ] NEW_LINE j -= 1 NEW_LINE DEDENT if ( prefix_sum == suffix_sum ) : NEW_LINE INDENT res = max ( res , prefix_sum ) NEW_LINE DEDENT DEDENT return res NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ - 2 , 5 , 3 , 1 , 2 , 6 , - 4 , 2 ] NEW_LINE n = len ( arr ) NEW_LINE print ( findMaxSum ( arr , n ) ) NEW_LINE DEDENT
def findMaxSum ( arr , n ) : NEW_LINE
preSum = [ 0 for i in range ( n ) ] NEW_LINE
suffSum = [ 0 for i in range ( n ) ] NEW_LINE
ans = - 10000000 NEW_LINE
preSum [ 0 ] = arr [ 0 ] NEW_LINE for i in range ( 1 , n ) : NEW_LINE INDENT preSum [ i ] = preSum [ i - 1 ] + arr [ i ] NEW_LINE DEDENT
suffSum [ n - 1 ] = arr [ n - 1 ] NEW_LINE if ( preSum [ n - 1 ] == suffSum [ n - 1 ] ) : NEW_LINE INDENT ans = max ( ans , preSum [ n - 1 ] ) NEW_LINE DEDENT for i in range ( n - 2 , - 1 , - 1 ) : NEW_LINE INDENT suffSum [ i ] = suffSum [ i + 1 ] + arr [ i ] NEW_LINE if ( suffSum [ i ] == preSum [ i ] ) : NEW_LINE INDENT ans = max ( ans , preSum [ i ] ) NEW_LINE DEDENT DEDENT return ans NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ - 2 , 5 , 3 , 1 , 2 , 6 , - 4 , 2 ] NEW_LINE n = len ( arr ) NEW_LINE print ( findMaxSum ( arr , n ) ) NEW_LINE DEDENT
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
def isSubsetSum ( set , n , sum ) : NEW_LINE
subset = ( [ [ False for i in range ( sum + 1 ) ] for i in range ( n + 1 ) ] ) NEW_LINE
for i in range ( n + 1 ) : NEW_LINE INDENT subset [ i ] [ 0 ] = True NEW_LINE DEDENT
for i in range ( 1 , sum + 1 ) : NEW_LINE INDENT subset [ 0 ] [ i ] = False NEW_LINE DEDENT
for i in range ( 1 , n + 1 ) : NEW_LINE INDENT for j in range ( 1 , sum + 1 ) : NEW_LINE INDENT if j < set [ i - 1 ] : NEW_LINE INDENT subset [ i ] [ j ] = subset [ i - 1 ] [ j ] NEW_LINE DEDENT if j >= set [ i - 1 ] : NEW_LINE INDENT subset [ i ] [ j ] = ( subset [ i - 1 ] [ j ] or subset [ i - 1 ] [ j - set [ i - 1 ] ] ) NEW_LINE DEDENT DEDENT DEDENT
for i in range ( n + 1 ) : NEW_LINE for j in range ( sum + 1 ) : NEW_LINE print ( subset [ i ] [ j ] , end = " ▁ " ) NEW_LINE print ( ) NEW_LINE return subset [ n ] [ sum ] NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT set = [ 3 , 34 , 4 , 12 , 5 , 2 ] NEW_LINE sum = 9 NEW_LINE n = len ( set ) NEW_LINE if ( isSubsetSum ( set , n , sum ) == True ) : NEW_LINE INDENT print ( " Found ▁ a ▁ subset ▁ with ▁ given ▁ sum " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No ▁ subset ▁ with ▁ given ▁ sum " ) NEW_LINE DEDENT DEDENT
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
SieveOfEratosthenes ( ) NEW_LINE
l = 3 NEW_LINE r = 9 NEW_LINE
c = ( sum [ r ] - sum [ l - 1 ] ) NEW_LINE
print ( " Count : " , c ) NEW_LINE
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
def isPowerOfTwo ( x ) : NEW_LINE
return ( x and ( not ( x & ( x - 1 ) ) ) ) NEW_LINE
if ( isPowerOfTwo ( 31 ) ) : NEW_LINE INDENT print ( ' Yes ' ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( ' No ' ) NEW_LINE DEDENT if ( isPowerOfTwo ( 64 ) ) : NEW_LINE INDENT print ( ' Yes ' ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( ' No ' ) NEW_LINE DEDENT
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
def printTetra ( n ) : NEW_LINE INDENT if ( n < 0 ) : NEW_LINE INDENT return ; NEW_LINE DEDENT DEDENT
first = 0 ; NEW_LINE second = 1 ; NEW_LINE third = 1 ; NEW_LINE fourth = 2 ; NEW_LINE
curr = 0 ; NEW_LINE if ( n == 0 ) : NEW_LINE INDENT print ( first ) ; NEW_LINE DEDENT elif ( n == 1 or n == 2 ) : NEW_LINE INDENT print ( second ) ; NEW_LINE DEDENT elif ( n == 3 ) : NEW_LINE INDENT print ( fourth ) ; NEW_LINE DEDENT else : NEW_LINE
for i in range ( 4 , n + 1 ) : NEW_LINE INDENT curr = first + second + third + fourth ; NEW_LINE first = second ; NEW_LINE second = third ; NEW_LINE third = fourth ; NEW_LINE fourth = curr ; NEW_LINE DEDENT print ( curr ) ; NEW_LINE
n = 10 ; NEW_LINE printTetra ( n ) ; NEW_LINE
def countWays ( n ) : NEW_LINE INDENT res = [ 0 ] * ( n + 2 ) NEW_LINE res [ 0 ] = 1 NEW_LINE res [ 1 ] = 1 NEW_LINE res [ 2 ] = 2 NEW_LINE for i in range ( 3 , n + 1 ) : NEW_LINE INDENT res [ i ] = res [ i - 1 ] + res [ i - 2 ] + res [ i - 3 ] NEW_LINE DEDENT return res [ n ] NEW_LINE DEDENT
n = 4 NEW_LINE print ( countWays ( n ) ) NEW_LINE
def maxTasks ( high , low , n ) : NEW_LINE
if ( n <= 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
return max ( high [ n - 1 ] + maxTasks ( high , low , ( n - 2 ) ) , low [ n - 1 ] + maxTasks ( high , low , ( n - 1 ) ) ) ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 5 ; NEW_LINE high = [ 3 , 6 , 8 , 7 , 6 ] NEW_LINE low = [ 1 , 5 , 4 , 5 , 3 ] NEW_LINE print ( maxTasks ( high , low , n ) ) ; NEW_LINE DEDENT
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
def printTetraRec ( n ) : NEW_LINE
if ( n == 0 ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT
if ( n == 1 or n == 2 ) : NEW_LINE INDENT return 1 ; NEW_LINE DEDENT
if ( n == 3 ) : NEW_LINE INDENT return 2 ; NEW_LINE DEDENT else : NEW_LINE INDENT return ( printTetraRec ( n - 1 ) + printTetraRec ( n - 2 ) + printTetraRec ( n - 3 ) + printTetraRec ( n - 4 ) ) ; NEW_LINE DEDENT
def printTetra ( n ) : NEW_LINE INDENT print ( printTetraRec ( n ) , end = " ▁ " ) ; NEW_LINE DEDENT
n = 10 ; NEW_LINE printTetra ( n ) ; NEW_LINE
/ * Returns the maximum among the 2 numbers * / NEW_LINE def max1 ( x , y ) : NEW_LINE INDENT return x if ( x > y ) else y ; NEW_LINE DEDENT
def maxTasks ( high , low , n ) : NEW_LINE
task_dp = [ 0 ] * ( n + 1 ) ; NEW_LINE
task_dp [ 0 ] = 0 ; NEW_LINE
task_dp [ 1 ] = high [ 0 ] ; NEW_LINE
for i in range ( 2 , n + 1 ) : NEW_LINE INDENT task_dp [ i ] = max ( high [ i - 1 ] + task_dp [ i - 2 ] , low [ i - 1 ] + task_dp [ i - 1 ] ) ; NEW_LINE DEDENT return task_dp [ n ] ; NEW_LINE
n = 5 ; NEW_LINE high = [ 3 , 6 , 8 , 7 , 6 ] ; NEW_LINE low = [ 1 , 5 , 4 , 5 , 3 ] ; NEW_LINE print ( maxTasks ( high , low , n ) ) ; NEW_LINE
n = 10 NEW_LINE k = 2 NEW_LINE print ( " Value ▁ of ▁ P ( " , n , " , ▁ " , k , " ) ▁ is ▁ " , permutationCoeff ( n , k ) ) NEW_LINE
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
def minOperations ( str , n ) : NEW_LINE
lastUpper = - 1 NEW_LINE firstLower = - 1 NEW_LINE
for i in range ( n - 1 , - 1 , - 1 ) : NEW_LINE INDENT if ( str [ i ] . isupper ( ) ) : NEW_LINE INDENT lastUpper = i NEW_LINE break NEW_LINE DEDENT DEDENT
for i in range ( n ) : NEW_LINE INDENT if ( str [ i ] . islower ( ) ) : NEW_LINE INDENT firstLower = i NEW_LINE break NEW_LINE DEDENT DEDENT
if ( lastUpper == - 1 or firstLower == - 1 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
countUpper = 0 NEW_LINE for i in range ( firstLower , n ) : NEW_LINE INDENT if ( str [ i ] . isupper ( ) ) : NEW_LINE INDENT countUpper += 1 NEW_LINE DEDENT DEDENT
countLower = 0 NEW_LINE for i in range ( lastUpper ) : NEW_LINE INDENT if ( str [ i ] . islower ( ) ) : NEW_LINE INDENT countLower += 1 NEW_LINE DEDENT DEDENT
return min ( countLower , countUpper ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT str = " geEksFOrGEekS " NEW_LINE n = len ( str ) NEW_LINE print ( minOperations ( str , n ) ) NEW_LINE DEDENT
def rainDayProbability ( a , n ) : NEW_LINE
count = a . count ( 1 ) NEW_LINE
m = count / n NEW_LINE return m NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = [ 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 ] NEW_LINE n = len ( a ) NEW_LINE print ( rainDayProbability ( a , n ) ) NEW_LINE DEDENT
def Series ( n ) : NEW_LINE INDENT sums = 0.0 NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE INDENT ser = 1 / ( i ** i ) NEW_LINE sums += ser NEW_LINE DEDENT return sums NEW_LINE DEDENT
n = 3 NEW_LINE res = round ( Series ( n ) , 5 ) NEW_LINE print ( res ) NEW_LINE
def ternarySearch ( l , r , key , ar ) : NEW_LINE INDENT if ( r >= l ) : NEW_LINE DEDENT
mid1 = l + ( r - l ) // 3 NEW_LINE mid2 = r - ( r - l ) // 3 NEW_LINE
if ( ar [ mid1 ] == key ) : NEW_LINE INDENT return mid1 NEW_LINE DEDENT if ( ar [ mid2 ] == key ) : NEW_LINE INDENT return mid2 NEW_LINE DEDENT
if ( key < ar [ mid1 ] ) : NEW_LINE
return ternarySearch ( l , mid1 - 1 , key , ar ) NEW_LINE elif ( key > ar [ mid2 ] ) : NEW_LINE
return ternarySearch ( mid2 + 1 , r , key , ar ) NEW_LINE else : NEW_LINE
return ternarySearch ( mid1 + 1 , mid2 - 1 , key , ar ) NEW_LINE
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
import numpy as np NEW_LINE
def prCharWithFreq ( str ) : NEW_LINE
' NEW_LINE INDENT n = len ( str ) NEW_LINE DEDENT
' NEW_LINE INDENT for i in range ( 0 , n ) : NEW_LINE INDENT freq [ ord ( str [ i ] ) - ord ( ' a ' ) ] += 1 NEW_LINE DEDENT DEDENT
for i in range ( 0 , n ) : NEW_LINE
if ( freq [ ord ( str [ i ] ) - ord ( ' a ' ) ] != 0 ) : NEW_LINE
print ( str [ i ] , freq [ ord ( str [ i ] ) - ord ( ' a ' ) ] , end = " ▁ " ) NEW_LINE
freq [ ord ( str [ i ] ) - ord ( ' a ' ) ] = 0 NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT str = " geeksforgeeks " ; NEW_LINE prCharWithFreq ( str ) ; NEW_LINE DEDENT ' NEW_LINE
MAX = 1000 NEW_LINE def checkHV ( arr , N , M ) : NEW_LINE
horizontal = True NEW_LINE vertical = True NEW_LINE
i = 0 NEW_LINE k = N - 1 NEW_LINE while ( i < N // 2 ) : NEW_LINE
for j in range ( M ) : NEW_LINE
if ( arr [ i ] [ j ] != arr [ k ] [ j ] ) : NEW_LINE INDENT horizontal = False NEW_LINE break NEW_LINE DEDENT i += 1 NEW_LINE k -= 1 NEW_LINE
i = 0 NEW_LINE k = M - 1 NEW_LINE while ( i < M // 2 ) : NEW_LINE
for j in range ( N ) : NEW_LINE
if ( arr [ i ] [ j ] != arr [ k ] [ j ] ) : NEW_LINE INDENT vertical = False NEW_LINE break NEW_LINE DEDENT i += 1 NEW_LINE k -= 1 NEW_LINE if ( not horizontal and not vertical ) : NEW_LINE print ( " NO " ) NEW_LINE elif ( horizontal and not vertical ) : NEW_LINE print ( " HORIZONTAL " ) NEW_LINE elif ( vertical and not horizontal ) : NEW_LINE print ( " VERTICAL " ) NEW_LINE else : NEW_LINE print ( " BOTH " ) NEW_LINE
mat = [ [ 1 , 0 , 1 ] , [ 0 , 0 , 0 ] , [ 1 , 0 , 1 ] ] NEW_LINE checkHV ( mat , 3 , 3 ) NEW_LINE
N = 4 NEW_LINE
def add ( A , B , C ) : NEW_LINE INDENT for i in range ( N ) : NEW_LINE INDENT for j in range ( N ) : NEW_LINE INDENT C [ i ] [ j ] = A [ i ] [ j ] + B [ i ] [ j ] NEW_LINE DEDENT DEDENT DEDENT
A = [ [ 1 , 1 , 1 , 1 ] , [ 2 , 2 , 2 , 2 ] , [ 3 , 3 , 3 , 3 ] , [ 4 , 4 , 4 , 4 ] ] NEW_LINE B = [ [ 1 , 1 , 1 , 1 ] , [ 2 , 2 , 2 , 2 ] , [ 3 , 3 , 3 , 3 ] , [ 4 , 4 , 4 , 4 ] ] NEW_LINE C = A [ : ] [ : ] NEW_LINE add ( A , B , C ) NEW_LINE print ( " Result ▁ matrix ▁ is " ) NEW_LINE for i in range ( N ) : NEW_LINE INDENT for j in range ( N ) : NEW_LINE INDENT print ( C [ i ] [ j ] , " ▁ " , end = ' ' ) NEW_LINE DEDENT print ( ) NEW_LINE DEDENT
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
def countSort ( arr ) : NEW_LINE
output = [ 0 for i in range ( len ( arr ) ) ] NEW_LINE
count = [ 0 for i in range ( 256 ) ] NEW_LINE
for i in arr : NEW_LINE INDENT count [ ord ( i ) ] += 1 NEW_LINE DEDENT
for i in range ( 256 ) : NEW_LINE INDENT count [ i ] += count [ i - 1 ] NEW_LINE DEDENT
for i in range ( len ( arr ) ) : NEW_LINE INDENT output [ count [ ord ( arr [ i ] ) ] - 1 ] = arr [ i ] NEW_LINE count [ ord ( arr [ i ] ) ] -= 1 NEW_LINE DEDENT
ans = [ " " for _ in arr ] NEW_LINE for i in range ( len ( arr ) ) : NEW_LINE INDENT ans [ i ] = output [ i ] NEW_LINE DEDENT return ans NEW_LINE
arr = " geeksforgeeks " NEW_LINE ans = countSort ( arr ) NEW_LINE print ( " Sorted ▁ character ▁ array ▁ is ▁ % ▁ s " % ( " " . join ( ans ) ) ) NEW_LINE
def binomialCoeff ( n , k ) : NEW_LINE
if k > n : NEW_LINE INDENT return 0 NEW_LINE DEDENT if k == 0 or k == n : NEW_LINE INDENT return 1 NEW_LINE DEDENT
return binomialCoeff ( n - 1 , k - 1 ) + binomialCoeff ( n - 1 , k ) NEW_LINE
n = 5 NEW_LINE k = 2 NEW_LINE print " Value ▁ of ▁ C ( % d , % d ) ▁ is ▁ ( % d ) " % ( n , k , binomialCoeff ( n , k ) ) NEW_LINE
def binomialCoeff ( n , k ) : NEW_LINE INDENT C = [ 0 for i in xrange ( k + 1 ) ] NEW_LINE DEDENT
C [ 0 ] = 1 NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE
j = min ( i , k ) NEW_LINE while ( j > 0 ) : NEW_LINE INDENT C [ j ] = C [ j ] + C [ j - 1 ] NEW_LINE j -= 1 NEW_LINE DEDENT return C [ k ] NEW_LINE
n = 5 NEW_LINE k = 2 NEW_LINE print " Value ▁ of ▁ C ( % d , % d ) ▁ is ▁ % d " % ( n , k , binomialCoeff ( n , k ) ) NEW_LINE
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
def power ( x , y ) : NEW_LINE INDENT if ( y == 0 ) : return 1 NEW_LINE elif ( int ( y % 2 ) == 0 ) : NEW_LINE INDENT return ( power ( x , int ( y / 2 ) ) * power ( x , int ( y / 2 ) ) ) NEW_LINE DEDENT else : NEW_LINE INDENT return ( x * power ( x , int ( y / 2 ) ) * power ( x , int ( y / 2 ) ) ) NEW_LINE DEDENT DEDENT
x = 2 ; y = 3 NEW_LINE print ( power ( x , y ) ) NEW_LINE
def power ( x , y ) : NEW_LINE INDENT if ( y == 0 ) : return 1 NEW_LINE temp = power ( x , int ( y / 2 ) ) NEW_LINE if ( y % 2 == 0 ) : NEW_LINE INDENT return temp * temp NEW_LINE DEDENT else : NEW_LINE INDENT if ( y > 0 ) : return x * temp * temp NEW_LINE else : return ( temp * temp ) / x NEW_LINE DEDENT DEDENT
x , y = 2 , - 3 NEW_LINE print ( ' % .6f ' % ( power ( x , y ) ) ) NEW_LINE
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
def maxIndexDiff ( arr , n ) : NEW_LINE INDENT maxDiff = - 1 NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT j = n - 1 NEW_LINE while ( j > i ) : NEW_LINE INDENT if arr [ j ] > arr [ i ] and maxDiff < ( j - i ) : NEW_LINE INDENT maxDiff = j - i NEW_LINE DEDENT j -= 1 NEW_LINE DEDENT DEDENT return maxDiff NEW_LINE DEDENT
arr = [ 9 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 18 , 0 ] NEW_LINE n = len ( arr ) NEW_LINE maxDiff = maxIndexDiff ( arr , n ) NEW_LINE print ( maxDiff ) NEW_LINE
def missingK ( a , k , n ) : NEW_LINE INDENT difference = 0 NEW_LINE ans = 0 NEW_LINE count = k NEW_LINE flag = 0 NEW_LINE DEDENT
for i in range ( 0 , n - 1 ) : NEW_LINE INDENT difference = 0 NEW_LINE DEDENT
if ( ( a [ i ] + 1 ) != a [ i + 1 ] ) : NEW_LINE
difference += ( a [ i + 1 ] - a [ i ] ) - 1 NEW_LINE
if ( difference >= count ) : NEW_LINE INDENT ans = a [ i ] + count NEW_LINE flag = 1 NEW_LINE break NEW_LINE DEDENT else : NEW_LINE INDENT count -= difference NEW_LINE DEDENT
if ( flag ) : NEW_LINE INDENT return ans NEW_LINE DEDENT else : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
a = [ 1 , 5 , 11 , 19 ] NEW_LINE
k = 11 NEW_LINE n = len ( a ) NEW_LINE
missing = missingK ( a , k , n ) NEW_LINE print ( missing ) NEW_LINE
def findRotations ( str ) : NEW_LINE
tmp = str + str NEW_LINE n = len ( str ) NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE
substring = tmp [ i : i + n ] NEW_LINE
if ( str == substring ) : NEW_LINE INDENT return i NEW_LINE DEDENT return n NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT str = " abc " NEW_LINE print ( findRotations ( str ) ) NEW_LINE DEDENT
def findKth ( arr , n , k ) : NEW_LINE INDENT missing = dict ( ) NEW_LINE count = 0 NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT missing [ arr [ i ] ] = 1 NEW_LINE DEDENT
maxm = max ( arr ) NEW_LINE minm = min ( arr ) NEW_LINE
for i in range ( minm + 1 , maxm ) : NEW_LINE
if ( i not in missing . keys ( ) ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT
if ( count == k ) : NEW_LINE INDENT return i NEW_LINE DEDENT
return - 1 NEW_LINE
arr = [ 2 , 10 , 9 , 4 ] NEW_LINE n = len ( arr ) NEW_LINE k = 5 NEW_LINE print ( findKth ( arr , n , k ) ) NEW_LINE
def waysToKAdjacentSetBits ( n , k , currentIndex , adjacentSetBits , lastBit ) : NEW_LINE
if ( currentIndex == n ) : NEW_LINE
if ( adjacentSetBits == k ) : NEW_LINE INDENT return 1 ; NEW_LINE DEDENT return 0 NEW_LINE noOfWays = 0 NEW_LINE
if ( lastBit == 1 ) : NEW_LINE
noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits + 1 , 1 ) ; NEW_LINE
noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits , 0 ) ; NEW_LINE elif ( lastBit != 1 ) : NEW_LINE noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits , 1 ) ; NEW_LINE noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits , 0 ) ; NEW_LINE return noOfWays ; NEW_LINE
n = 5 ; k = 2 ; NEW_LINE
totalWays = ( waysToKAdjacentSetBits ( n , k , 1 , 0 , 1 ) + waysToKAdjacentSetBits ( n , k , 1 , 0 , 0 ) ) ; NEW_LINE print ( " Number ▁ of ▁ ways ▁ = " , totalWays ) ; NEW_LINE
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
def findRepeatFirstN2 ( s ) : NEW_LINE
p = - 1 NEW_LINE for i in range ( len ( s ) ) : NEW_LINE INDENT for j in range ( i + 1 , len ( s ) ) : NEW_LINE INDENT if ( s [ i ] == s [ j ] ) : NEW_LINE INDENT p = i NEW_LINE break NEW_LINE DEDENT DEDENT if ( p != - 1 ) : NEW_LINE INDENT break NEW_LINE DEDENT DEDENT return p NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT str = " geeksforgeeks " NEW_LINE pos = findRepeatFirstN2 ( str ) NEW_LINE if ( pos == - 1 ) : NEW_LINE INDENT print ( " Not ▁ found " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( str [ pos ] ) NEW_LINE DEDENT DEDENT
def possibleStrings ( n , r , b , g ) : NEW_LINE
fact = [ 0 for i in range ( n + 1 ) ] NEW_LINE fact [ 0 ] = 1 NEW_LINE for i in range ( 1 , n + 1 , 1 ) : NEW_LINE INDENT fact [ i ] = fact [ i - 1 ] * i NEW_LINE DEDENT
left = n - ( r + g + b ) NEW_LINE sum = 0 NEW_LINE
for i in range ( 0 , left + 1 , 1 ) : NEW_LINE INDENT for j in range ( 0 , left - i + 1 , 1 ) : NEW_LINE INDENT k = left - ( i + j ) NEW_LINE DEDENT DEDENT
sum = ( sum + fact [ n ] / ( fact [ i + r ] * fact [ j + b ] * fact [ k + g ] ) ) NEW_LINE
return sum NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 4 NEW_LINE r = 2 NEW_LINE b = 0 NEW_LINE g = 1 NEW_LINE print ( int ( possibleStrings ( n , r , b , g ) ) ) NEW_LINE DEDENT
def remAnagram ( str1 , str2 ) : NEW_LINE
count1 = [ 0 ] * CHARS NEW_LINE count2 = [ 0 ] * CHARS NEW_LINE
i = 0 NEW_LINE while i < len ( str1 ) : NEW_LINE INDENT count1 [ ord ( str1 [ i ] ) - ord ( ' a ' ) ] += 1 NEW_LINE i += 1 NEW_LINE DEDENT
i = 0 NEW_LINE while i < len ( str2 ) : NEW_LINE INDENT count2 [ ord ( str2 [ i ] ) - ord ( ' a ' ) ] += 1 NEW_LINE i += 1 NEW_LINE DEDENT
result = 0 NEW_LINE for i in range ( 26 ) : NEW_LINE INDENT result += abs ( count1 [ i ] - count2 [ i ] ) NEW_LINE DEDENT return result NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT str1 = " bcadeh " NEW_LINE str2 = " hea " NEW_LINE print ( remAnagram ( str1 , str2 ) ) NEW_LINE DEDENT
def printPath ( res , nThNode , kThNode ) : NEW_LINE
if kThNode > nThNode : NEW_LINE INDENT return NEW_LINE DEDENT
res . append ( kThNode ) NEW_LINE
for i in range ( 0 , len ( res ) ) : NEW_LINE INDENT print ( res [ i ] , end = " ▁ " ) NEW_LINE DEDENT print ( ) NEW_LINE
printPath ( res [ : ] , nThNode , kThNode * 2 ) NEW_LINE
printPath ( res [ : ] , nThNode , kThNode * 2 + 1 ) NEW_LINE
def printPathToCoverAllNodeUtil ( nThNode ) : NEW_LINE
res = [ ] NEW_LINE
printPath ( res , nThNode , 1 ) NEW_LINE
nThNode = 7 NEW_LINE
printPathToCoverAllNodeUtil ( nThNode ) NEW_LINE
def shortestLength ( n , x , y ) : NEW_LINE INDENT answer = 0 NEW_LINE DEDENT
i = 0 NEW_LINE while n > 0 : NEW_LINE
if ( x [ i ] + y [ i ] > answer ) : NEW_LINE INDENT answer = x [ i ] + y [ i ] NEW_LINE DEDENT i += 1 NEW_LINE n -= 1 NEW_LINE
print ( " Length ▁ - > ▁ " + str ( answer ) ) NEW_LINE print ( " Path ▁ - > ▁ " + " ( ▁ 1 , ▁ " + str ( answer ) + " ▁ ) " + " and ▁ ( ▁ " + str ( answer ) + " , ▁ 1 ▁ ) " ) NEW_LINE
n = 4 NEW_LINE
x = [ 1 , 4 , 2 , 1 ] NEW_LINE y = [ 4 , 1 , 1 , 2 ] NEW_LINE shortestLength ( n , x , y ) NEW_LINE
def FindPoints ( x1 , y1 , x2 , y2 , x3 , y3 , x4 , y4 ) : NEW_LINE
x5 = max ( x1 , x3 ) NEW_LINE y5 = max ( y1 , y3 ) NEW_LINE
x6 = min ( x2 , x4 ) NEW_LINE y6 = min ( y2 , y4 ) NEW_LINE
if ( x5 > x6 or y5 > y6 ) : NEW_LINE INDENT print ( " No ▁ intersection " ) NEW_LINE return NEW_LINE DEDENT print ( " ( " , x5 , " , ▁ " , y5 , " ) ▁ " , end = " ▁ " ) NEW_LINE print ( " ( " , x6 , " , ▁ " , y6 , " ) ▁ " , end = " ▁ " ) NEW_LINE
x7 = x5 NEW_LINE y7 = y6 NEW_LINE print ( " ( " , x7 , " , ▁ " , y7 , " ) ▁ " , end = " ▁ " ) NEW_LINE
x8 = x6 NEW_LINE y8 = y5 NEW_LINE print ( " ( " , x8 , " , ▁ " , y8 , " ) ▁ " ) NEW_LINE
x1 = 0 NEW_LINE y1 = 0 NEW_LINE x2 = 10 NEW_LINE y2 = 8 NEW_LINE
x3 = 2 NEW_LINE y3 = 3 NEW_LINE x4 = 7 NEW_LINE y4 = 9 NEW_LINE
FindPoints ( x1 , y1 , x2 , y2 , x3 , y3 , x4 , y4 ) NEW_LINE
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
def isPalindrome ( num ) : NEW_LINE INDENT reverse_num = 0 NEW_LINE DEDENT
temp = num NEW_LINE while ( temp != 0 ) : NEW_LINE INDENT remainder = temp % 10 NEW_LINE reverse_num = reverse_num * 10 + remainder NEW_LINE temp = int ( temp / 10 ) NEW_LINE DEDENT
if ( reverse_num == num ) : NEW_LINE INDENT return True NEW_LINE DEDENT return False NEW_LINE
def isOddLength ( num ) : NEW_LINE INDENT count = 0 NEW_LINE while ( num > 0 ) : NEW_LINE INDENT num = int ( num / 10 ) NEW_LINE count += 1 NEW_LINE DEDENT if ( count % 2 != 0 ) : NEW_LINE INDENT return True NEW_LINE DEDENT return False NEW_LINE DEDENT
def sumOfAllPalindrome ( L , R ) : NEW_LINE INDENT sum = 0 NEW_LINE if ( L <= R ) : NEW_LINE INDENT for i in range ( L , R + 1 , 1 ) : NEW_LINE DEDENT DEDENT
if ( isPalindrome ( i ) and isOddLength ( i ) ) : NEW_LINE INDENT sum += i NEW_LINE DEDENT return sum NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT L = 110 NEW_LINE R = 1130 NEW_LINE print ( sumOfAllPalindrome ( L , R ) ) NEW_LINE DEDENT
def calculateAlternateSum ( n ) : NEW_LINE INDENT if ( n <= 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT fibo = [ 0 ] * ( n + 1 ) NEW_LINE fibo [ 0 ] = 0 NEW_LINE fibo [ 1 ] = 1 NEW_LINE DEDENT
sum = pow ( fibo [ 0 ] , 2 ) + pow ( fibo [ 1 ] , 2 ) NEW_LINE
for i in range ( 2 , n + 1 ) : NEW_LINE INDENT fibo [ i ] = fibo [ i - 1 ] + fibo [ i - 2 ] NEW_LINE DEDENT
if ( i % 2 == 0 ) : NEW_LINE INDENT sum -= fibo [ i ] NEW_LINE DEDENT
else : NEW_LINE INDENT sum += fibo [ i ] NEW_LINE DEDENT
return sum NEW_LINE
n = 8 NEW_LINE
print ( " Alternating ▁ Fibonacci ▁ Sum ▁ upto ▁ " , n , " ▁ terms : ▁ " , calculateAlternateSum ( n ) ) NEW_LINE
def getValue ( n ) : NEW_LINE INDENT i = 0 ; NEW_LINE k = 1 ; NEW_LINE while ( i < n ) : NEW_LINE INDENT i = i + k ; NEW_LINE k = k * 2 ; NEW_LINE DEDENT return int ( k / 2 ) ; NEW_LINE DEDENT
n = 9 ; NEW_LINE
print ( getValue ( n ) ) ; NEW_LINE
n = 1025 ; NEW_LINE
print ( getValue ( n ) ) ; NEW_LINE
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
def solve ( a , n ) : NEW_LINE INDENT max1 = - sys . maxsize - 1 NEW_LINE for i in range ( 0 , n , 1 ) : NEW_LINE INDENT for j in range ( 0 , n , 1 ) : NEW_LINE INDENT if ( abs ( a [ i ] - a [ j ] ) > max1 ) : NEW_LINE INDENT max1 = abs ( a [ i ] - a [ j ] ) NEW_LINE DEDENT DEDENT DEDENT return max1 NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ - 1 , 2 , 3 , - 4 , - 10 , 22 ] NEW_LINE size = len ( arr ) NEW_LINE print ( " Largest ▁ gap ▁ is ▁ : " , solve ( arr , size ) ) NEW_LINE DEDENT
def solve ( a , n ) : NEW_LINE INDENT min1 = a [ 0 ] NEW_LINE max1 = a [ 0 ] NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT if ( a [ i ] > max1 ) : NEW_LINE INDENT max1 = a [ i ] NEW_LINE DEDENT if ( a [ i ] < min1 ) : NEW_LINE INDENT min1 = a [ i ] NEW_LINE DEDENT DEDENT return abs ( min1 - max1 ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ - 1 , 2 , 3 , 4 , - 10 ] NEW_LINE size = len ( arr ) NEW_LINE print ( " Largest ▁ gap ▁ is ▁ : ▁ " , solve ( arr , size ) ) NEW_LINE DEDENT
def minElements ( arr , n ) : NEW_LINE
halfSum = 0 NEW_LINE for i in range ( n ) : NEW_LINE INDENT halfSum = halfSum + arr [ i ] NEW_LINE DEDENT halfSum = int ( halfSum / 2 ) NEW_LINE
arr . sort ( reverse = True ) NEW_LINE res = 0 NEW_LINE curr_sum = 0 NEW_LINE for i in range ( n ) : NEW_LINE INDENT curr_sum += arr [ i ] NEW_LINE res += 1 NEW_LINE DEDENT
if curr_sum > halfSum : NEW_LINE INDENT return res NEW_LINE DEDENT return res NEW_LINE
arr = [ 3 , 1 , 7 , 1 ] NEW_LINE n = len ( arr ) NEW_LINE print ( minElements ( arr , n ) ) NEW_LINE
def minCost ( N , P , Q ) : NEW_LINE
cost = 0 NEW_LINE
while ( N > 0 ) : NEW_LINE INDENT if ( N & 1 ) : NEW_LINE INDENT cost += P NEW_LINE N -= 1 NEW_LINE DEDENT else : NEW_LINE INDENT temp = N // 2 ; NEW_LINE DEDENT DEDENT
if ( temp * P > Q ) : NEW_LINE INDENT cost += Q NEW_LINE DEDENT
else : NEW_LINE INDENT cost += P * temp NEW_LINE DEDENT N //= 2 NEW_LINE return cost NEW_LINE
N = 9 NEW_LINE P = 5 NEW_LINE Q = 1 NEW_LINE print ( minCost ( N , P , Q ) ) NEW_LINE
